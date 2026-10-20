# Anthropic Performance Take-Home — Independent Attempt

**Independent attempt at Anthropic's publicly released retired take-home.
Not affiliated with Anthropic.**

This is Rene Turcios's original optimization work against the public simulator,
written in this session. No published solutions were consulted or copied.

## Result

| Metric | Cycles |
|---|---|
| Baseline (upstream starter, unmodified) | 147,734 |
| **Best correct (this attempt)** | **2,819** |
| 1487 email threshold | not beaten |

- Speedup over baseline: **52.4x**.
- Passes the harness speed tests `< 147734` and `< 18532`; does not pass `< 2164`.
- Correctness: the repo's `CorrectnessTests` (8 unseeded random runs) pass, plus
  additional multi-seed checks during development. A fast wrong answer was
  rejected during development (see OPTIMIZATION_LOG.md); only correct code is
  reported here.

## Layout

- `upstream/` — pristine clone of `github.com/anthropics/original_performance_takehome`
  (original files and license intact; untouched).
- `work/` — development area:
  - `perf_takehome.py` — the repo's file with only the `KernelBuilder` class
    replaced by the optimized version (harness, simulator, tests untouched).
  - `mykernel.py` — the optimized kernel builder (also kept standalone).
  - `problem.py`, `tests/` — unchanged from upstream.
- `OPTIMIZATION_LOG.md` — every attempt, measured cycles, failures, and dead ends.
- `README.md` — this file.

## How to run and verify

```bash
cd work
python3 tests/submission_tests.py   # runs correctness + speed thresholds
```

The harness reports `CYCLES: 2819` and `Speedup over baseline: 52.4...`.
`CorrectnessTests` must show `OK`. (The `test_*_speed` "failures" are the
harness's strict `<` thresholds you haven't beaten — expected; the important
part is correctness `OK` plus your measured number.)

## Main ideas

1. **Batch-resident vectors.** Values and level-relative indices live in scratch
   for all 16 rounds; only values are written back. Indices are maintained as
   level-relative `j = idx - (2^d - 1)` since depth `d = r mod 11` is static,
   turning the wrap into a depth-10-only zeroing step and the descent into
   branchless `j' = 2*j + (val & 1)`.
2. **Hash fusion.** `(a + C) + (a << k) = a*(2^k+1) + C` (mod 2^32) via the
   `multiply_add` valu slot, cutting the 6-stage `myhash` from 18 to 12 slots.
3. **Gather elision at shallow depths.** Depth-0 rounds: 1 scalar load +
   vbroadcast (all lanes at root). Depth-1: 2 scalar loads + per-lane `vselect`.
4. **17-stage software pipeline** over the 512 (round, vector) items with a
   greedy VLIW packer, so independent gather/hash/index work from neighboring
   items shares bundles under the slot limits (alu 12, valu 6, load 2, flow 1).

## Honest assessment

The per-item floor is 8 random gather loads = 4 load-engine cycles; 512 items
puts any straightforward vectorization above ~2048 cycles before overhead, so
1487 needs a structural insight beyond this approach (see log). Documented miss,
honest numbers.
