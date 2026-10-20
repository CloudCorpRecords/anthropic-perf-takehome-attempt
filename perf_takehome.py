"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


"""
Optimized kernel for Anthropic's publicly released performance take-home.

Independent attempt written from scratch against the simulator's documented
rules. Strategy:

- Keep the whole batch (values + level-relative indices j) resident in
  scratch across all 16 rounds; only values are written back at the end.
- Vectorize across VLEN=8 lanes with the valu engine. Fuse the (+,+,<<)
  hash stages into single multiply_add slots:
      (a + C) + (a << k)  ==  a * (2^k + 1) + C   (mod 2^32)
  cutting the 6-stage hash from 18 to 12 valu slots (verified numerically).
- The tree walk descends exactly one level per round, so depth d = r mod 11
  is known statically. The wrap-to-root select is only needed on max-depth
  rounds (replaced by a zeroing vbroadcast); other rounds use the branchless
  update j' = 2*j + (val & 1), derived from idx' = 2*idx + 1 + (val&1).
- Rounds at depth 0 (all lanes at the root) use 1 scalar load + vbroadcast
  instead of 256 gathers. Rounds at depth 1 (2 distinct nodes) use 2 scalar
  loads + a per-lane vselect instead of gathers.
- Software-pipeline the 512 (round, vector) work items across an 18-stage
  diagonal so independent work from neighboring items shares VLIW bundles.

Not affiliated with Anthropic.
"""

from problem import SLOT_LIMITS, VLEN, SCRATCH_SIZE, DebugInfo

# ---------------- scratch layout ----------------
JBASE = 0       # 32 j-vectors (level-relative indices), 256 words
VBASE = 256     # 32 value vectors, 256 words
CBASE = 512     # 14 const vectors (112) + 2 node vectors (16) = 128 words
TBASE = 640     # 18 temp banks x 32 words = 576 words
SBASE = 1216    # scalar scratch

NBANK = 17
BANKSZ = 32  # [0:8] addr, [8:16] node, [16:24] T1, [24:32] T2

CONSTS = [
    ("H1a", 0x7ED55D16),
    ("C4097", 4097),
    ("H2a", 0xC761C23C),
    ("C19", 19),
    ("H3a", 0x165667B1),
    ("C33", 33),
    ("H4a", 0xD3A2646C),
    ("C9", 9),
    ("H5a", 0xFD7046C5),
    ("H6a", 0xB55A4F09),
    ("C16v", 16),
    ("ONE", 1),
    ("TWOV", 2),
    ("ZERO_V", 0),
]
CVEC = {name: CBASE + 8 * i for i, (name, _) in enumerate(CONSTS)}

NSTAGE = 17  # pipeline depth (stages 0..16)


class VLIWPacker:
    """Greedy VLIW packer.

    wave() appends mutually-independent slots, flushing to a new bundle when
    an engine would exceed its slot limit. barrier() forces the current
    partial bundle out, separating dependent operations.
    """

    def __init__(self):
        self.instrs = []
        self.cur = {}

    def wave(self, slots):
        for eng, slot in slots:
            lst = self.cur.get(eng)
            if lst is None:
                self.cur[eng] = [slot]
            elif len(lst) >= SLOT_LIMITS[eng]:
                self.barrier()
                self.cur[eng] = [slot]
            else:
                lst.append(slot)

    def barrier(self):
        if self.cur:
            self.instrs.append({e: s for e, s in self.cur.items() if s})
            self.cur = {}

    def finish(self):
        self.barrier()
        return self.instrs


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self._sptr = SBASE
        self._scalars = {}

    def debug_info(self):
        return DebugInfo(scratch_map={})

    def scalar(self, name):
        if name not in self._scalars:
            addr = self._sptr
            self._sptr += 1
            assert self._sptr <= SCRATCH_SIZE, "out of scalar scratch"
            self._scalars[name] = addr
        return self._scalars[name]

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        assert batch_size % VLEN == 0, "batch must be a multiple of VLEN"
        nvec = batch_size // VLEN
        maxdepth = forest_height

        pk = VLIWPacker()

        FB = self.scalar("FB")
        EIGHT = self.scalar("EIGHT")
        ZERO_S = self.scalar("ZERO_S")
        MD_S = self.scalar("MD_S")
        VP = self.scalar("VP")
        AC = self.scalar("AC")
        T4 = self.scalar("T4")
        T6 = self.scalar("T6")
        NTMP_S = self.scalar("NTMP_S")   # depth-0 node value
        N0_S = self.scalar("N0_S")       # depth-1 node values
        N1_S = self.scalar("N1_S")
        ONE_S = self.scalar("ONE_S")
        CD = {d: self.scalar(f"CD{d}") for d in range(maxdepth + 1)}
        ST = [self.scalar(f"ST{i}") for i in range(len(CONSTS))]

        def const_waves(addr, val):
            pk.wave([("load", ("const", addr, val))])

        # pause for the local debug harness (yield matching)
        pk.wave([("flow", ("pause",))])
        pk.barrier()

        # FB = mem[4]
        const_waves(T4, 4)
        pk.barrier()
        pk.wave([("load", ("load", FB, T4))])
        pk.barrier()
        for name, val in [("EIGHT", 8), ("ZERO_S", 0), ("ONE_S", 1)]:
            const_waves(self.scalar(name), val)
        pk.barrier()
        for d in range(maxdepth + 1):
            const_waves(CD[d], (1 << d) - 1)
        pk.barrier()
        for i, (name, val) in enumerate(CONSTS):
            pk.wave([("load", ("const", ST[i], val))])
        pk.barrier()
        for i, (name, _) in enumerate(CONSTS):
            pk.wave([("valu", ("vbroadcast", CVEC[name], ST[i]))])
        pk.barrier()

        # j vectors = 0
        for k in range(nvec):
            pk.wave([("valu", ("vbroadcast", JBASE + 8 * k, ZERO_S))])
        pk.barrier()

        # value vectors = vload from mem[VP + 8k]
        const_waves(T6, 6)
        pk.barrier()
        pk.wave([("load", ("load", VP, T6))])
        pk.barrier()
        pk.wave([("alu", ("+", AC, VP, ZERO_S))])
        pk.barrier()
        for k in range(nvec):
            pk.wave([
                ("load", ("vload", VBASE + 8 * k, AC)),
                ("alu", ("+", AC, AC, EIGHT)),
            ])
            pk.barrier()

        # vector homes for depth-1 nodes (reused each depth-1 round)
        NV0 = CBASE + 8 * len(CONSTS)       # 2 extra vector slots
        NV1 = NV0 + 8
        assert NV1 + 8 <= TBASE, "const area overflow"

        # ---------- main pipeline ----------
        total_items = rounds * nvec

        def subwave(p, r, k, d, flavor):
            s = r * nvec + k
            B = TBASE + BANKSZ * (s % NBANK)
            JV = JBASE + 8 * k
            VV = VBASE + 8 * k
            AD = B + 0
            ND = B + 8
            T1 = B + 16
            T2 = B + 24
            if flavor == "S":
                if p == 0:
                    pk.wave([("alu", ("+", AD + j, JV + j, MD_S))
                             for j in range(VLEN)])
                elif p == 1:
                    pk.wave([("load", ("load", ND + j, AD + j))
                             for j in range(VLEN)])
                xor_p, base = 2, 0
            elif flavor == "Z":  # depth 0: node = tree[0] for all lanes
                if p == 1:
                    pk.wave([("valu", ("vbroadcast", ND, NTMP_S))])
                xor_p, base = 2, 0
            else:  # flavor "O": depth 1, two nodes + vselect (17 stages)
                if p == 0:
                    pk.wave([("valu", ("==", T1, JV, CVEC["ZERO_V"]))])
                elif p == 1:
                    pk.wave([("flow", ("vselect", ND, T1, NV0, NV1))])
                xor_p, base = 2, 0
            # xor + hash + parity + j-update, shifted by `base`
            q = p - base
            if q == xor_p:
                pk.wave([("valu", ("^", VV, VV, ND))])
            elif q == xor_p + 1:
                pk.wave([("valu", ("multiply_add", VV, VV,
                                  CVEC["C4097"], CVEC["H1a"]))])
            elif q == xor_p + 2:
                pk.wave([("valu", ("^", T1, VV, CVEC["H2a"]))])
            elif q == xor_p + 3:
                pk.wave([("valu", (">>", T2, VV, CVEC["C19"]))])
            elif q == xor_p + 4:
                pk.wave([("valu", ("^", VV, T1, T2))])
            elif q == xor_p + 5:
                pk.wave([("valu", ("multiply_add", VV, VV,
                                  CVEC["C33"], CVEC["H3a"]))])
            elif q == xor_p + 6:
                pk.wave([("valu", ("+", T1, VV, CVEC["H4a"]))])
            elif q == xor_p + 7:
                pk.wave([("valu", ("<<", T2, VV, CVEC["C9"]))])
            elif q == xor_p + 8:
                pk.wave([("valu", ("^", VV, T1, T2))])
            elif q == xor_p + 9:
                pk.wave([("valu", ("multiply_add", VV, VV,
                                  CVEC["C9"], CVEC["H5a"]))])
            elif q == xor_p + 10:
                pk.wave([("valu", ("^", T1, VV, CVEC["H6a"]))])
            elif q == xor_p + 11:
                pk.wave([("valu", (">>", T2, VV, CVEC["C16v"]))])
            elif q == xor_p + 12:
                pk.wave([("valu", ("^", VV, T1, T2))])
            elif q == xor_p + 13:
                pk.wave([("valu", ("&", T1, VV, CVEC["ONE"]))])
            elif q == xor_p + 14:
                if d == maxdepth:
                    pk.wave([("valu", ("vbroadcast", JV, ZERO_S))])
                else:
                    pk.wave([("valu", ("multiply_add", JV, JV,
                                      CVEC["TWOV"], T1))])

        for t in range(total_items + NSTAGE):
            if t < total_items and t % nvec == 0:
                r = t // nvec
                d = r % (maxdepth + 1)
                if d == 0:
                    # NTMP_S = mem[FB + 0]
                    pk.wave([("alu", ("+", AC, FB, ZERO_S))])
                    pk.barrier()
                    pk.wave([("load", ("load", NTMP_S, AC))])
                    pk.barrier()
                elif d == 1:
                    # N0_S = mem[FB+1], N1_S = mem[FB+2]; broadcast to NV0/NV1
                    pk.wave([("alu", ("+", AC, FB, CD[1]))])
                    pk.barrier()
                    pk.wave([("load", ("load", N0_S, AC))])
                    pk.barrier()
                    pk.wave([("alu", ("+", AC, AC, ONE_S))])
                    pk.barrier()
                    pk.wave([("load", ("load", N1_S, AC))])
                    pk.barrier()
                    pk.wave([("valu", ("vbroadcast", NV0, N0_S))])
                    pk.wave([("valu", ("vbroadcast", NV1, N1_S))])
                    pk.barrier()
                else:
                    # MD_S = FB + (2^d - 1) for scalar gathers
                    pk.wave([("alu", ("+", MD_S, FB, CD[d]))])
                    pk.barrier()
            # Emit sub-waves in packing-friendly order: valu-heavy stages first
            # so the load engine's bundles get filled with valu/alu slots.
            # (Stages are from independent items, so order is free.)
            order = list(range(2, NSTAGE)) + [0, 1]
            for p in order:
                s = t - p
                if 0 <= s < total_items:
                    r = s // nvec
                    k = s % nvec
                    d = r % (maxdepth + 1)
                    flavor = "Z" if d == 0 else ("O" if d == 1 else "S")
                    subwave(p, r, k, d, flavor)
            pk.barrier()

        # ---------- write back values ----------
        pk.wave([("alu", ("+", AC, VP, ZERO_S))])
        pk.barrier()
        for k in range(nvec):
            pk.wave([
                ("store", ("vstore", AC, VBASE + 8 * k)),
                ("alu", ("+", AC, AC, EIGHT)),
            ])
            pk.barrier()

        pk.wave([("flow", ("pause",))])
        pk.barrier()

        self.instrs = pk.finish()


BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
