# Verification methodology

## Reference model strategy

The quantized network has one executable specification: `model/golden.py`, a
pure-integer NumPy implementation whose every operation maps 1:1 onto RTL
hardware (same accumulation structure, same rounding, same clamps). PyTorch is
only used to *train* weights; correctness is defined by the golden model, so
there is no floating-point ambiguity anywhere in the check path.

## Self-checking testbench (tb/tb_accel.sv)

Per simulation run, driven by `sim/run_sim.py` (which generates all
expected-value files from the golden model before compiling):

| Check | What it catches | Granularity |
|---|---|---|
| Hidden activation compare | array mapping, tiling order, bias injection, requantize arithmetic | every requantized value, bit-exact (`!==`, X-propagation counts as failure) |
| Logit compare | end-to-end datapath | all 10 INT32 logits per image, bit-exact |
| Prediction compare | (redundant with logits; sanity) | argmax vs golden argmax + true label |
| X-check on logit stream | uninitialized/unknown propagation | `$finish` on any X |
| Drain/protocol assertion | control timing (a block fully drains before its preload) | every cycle |

Checking *intermediate* values, not just predictions, matters: an argmax
agrees with the golden model 10% of the time by luck, per-logit checks
constrain ~2³² states each, and activation checks localize a failure to a
specific layer/output-block/tile pass. During bring-up, this is exactly how
two real bugs were found and localized in minutes (see Bug log below).

## Regression runs

| Run | Purpose |
|---|---|
| 200 MNIST images, batch 16 | main regression + accuracy measurement |
| 16 images, batch 1/2/4/8 | batching/utilization sweep + tail-batch handling |
| 4 synthetic stress images, batch 1 | corner cases: all-zero image, all-127 (drives requant saturation), checkerboard, ramp |
| 8×8 array, 200 images | parameterization proof + scaling measurement |
| Array-level smoke test | bare-array dataflow with ±127/−128 extremes and zero vectors |

## Functional coverage

Icarus Verilog has no covergroups, so coverage is implemented as explicit
counters sampled every cycle, reported per run and unioned across the
regression list: weight bins (zero / positive / negative / ±127 extremes),
activation bins (zero / mid / max-127), requantizer bins (ReLU-clamp /
mid-range / saturate-high), accumulator magnitude bins (4 decades), partial-sum
sign bins, and batch-size extremes. **19/19 bins hit (100%)** — the saturation
and batch bins require the stress run, which is the point of having it.

## Bug log (found by this methodology)

1. **Weight-load off-by-one** — synchronous weight-memory read lags the
   address pointer by a cycle; every PE row was loaded with the previous
   row's word. Caught by: activation mismatches on nearly all output blocks;
   localized by comparing block-0 accumulators against golden partials.
2. **Unsigned-concatenation requantize bug** — the rounding constant was a
   concatenation (always unsigned in Verilog), which forced the whole
   `acc*M + round` expression unsigned and zero-extended negative
   accumulators. Caught by: accumulators bit-exact but requantized outputs
   wrong only for negative accs (ReLU cases returned 127 instead of 0).
3. **`done`-latch race** — `done` stays asserted between runs; the testbench's
   `wait (done)` for run 2 fell through instantly. Caught by: run-2 cycle
   counter reading zero.

The trace infrastructure (per-cycle JSON dump of every PE) doubles as a
debugging tool: bug 1's diagnosis came from grepping requantize events out of
the trace and diffing against golden layer-0 partial sums.

## Re-verification after the performance rework

The weight double-buffering + pipelined-requantizer rework (swap tokens,
overlapped drains, batch-modulo accumulator counters, per-column first-tile
flags) changed control timing everywhere but is *supposed* to be
arithmetically invisible. Because every expected value is bit-exact and
independent of schedule, the entire regression list re-ran unchanged and
passed on the first attempt — which is precisely the payoff of checking
against an executable spec rather than hand-derived waveforms.
