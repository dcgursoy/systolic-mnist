# Accelerator architecture

A weight-stationary systolic array executing a quantized 784→32→16→10 MLP
end-to-end in hardware: all three matrix multiplies, bias handling,
requantization, and ReLU. The testbench only loads pixels and reads logits.

## Top level

```mermaid
flowchart LR
    subgraph accel_top
        direction LR
        WMEM["weight mem<br/>6448 x 32b<br/>(packed tiles)"]
        BMEM["bias mem<br/>15 x 128b"]
        ACT0["act banks parity 0<br/>4 x (16 x 196) INT8"]
        ACT1["act banks parity 1<br/>4 x (16 x 196) INT8"]
        SKEW["skew buffer<br/>(row r delayed r cyc)"]
        ARR["4 x 4 PE array<br/>weight-stationary"]
        ACC["4 x acc_bank<br/>INT32, bias inject"]
        REQ["4 x requantize<br/>acc*M >> s, clamp"]
        CTRL["controller FSM<br/>layer / block / tile loops"]
    end
    IMG[image port] --> ACT0
    WMEM -- "row per cycle" --> ARR
    ACT0 --> SKEW
    ACT1 --> SKEW
    SKEW --> ARR
    ARR -- "column psums" --> ACC
    BMEM --> ACC
    ACC --> REQ
    REQ -- "hidden layers" --> ACT1
    REQ -- "hidden layers" --> ACT0
    ACC -- "final layer (raw INT32)" --> LOG[logit stream]
    CTRL -.-> WMEM & BMEM & SKEW & ARR & ACC & REQ
```

## The PE array

Each PE holds one INT8 weight and performs one INT8×INT8 MAC per cycle into
the INT32 partial sum flowing through it:

```
            x[0]──▶ PE00 ─▶ PE01 ─▶ PE02 ─▶ PE03        activations flow east
       x[1]──────▶ PE10 ─▶ PE11 ─▶ PE12 ─▶ PE13        (skewed by 1 cycle/row)
  x[2]───────────▶ PE20 ─▶ PE21 ─▶ PE22 ─▶ PE23
x[3]─────────────▶ PE30 ─▶ PE31 ─▶ PE32 ─▶ PE33
                    │       │       │       │            partial sums flow south
                    ▼       ▼       ▼       ▼
                  acc0    acc1    acc2    acc3           per-column accumulators
```

PE(r,c) holds `W[out = c][in = r]`: **column c owns output neuron c** of the
current 4×4 tile. A full column sum takes N cycles to cascade down, so column
c's result for a vector emerges `N + c` cycles after that vector's first
element enters the west edge — outputs are naturally skewed, and the
controller's per-column valid pipelines account for it.

Weights load through a separate port, one row per cycle (N+1 cycles per tile
including the synchronous-read address lead). Because the weight memory is
packed offline in exactly the loop order the controller executes
(layer → output block → input tile → row), the address generator is a single
incrementing pointer.

## Tiling: mapping a 784×32 matmul onto a 4×4 array

Real accelerators are always much smaller than the workload; the schedule
below is the standard output-block / reduction-tile decomposition:

```
for layer  l in 0..2:                     # 784x32, 32x16, 16x10(pad 12)
  for output block j in 0..out/4-1:      # 4 output neurons at a time
    for input tile i in 0..in/4-1:       # reduce over input in chunks of 4
      load W[l][j*4..j*4+3][i*4..i*4+3]  # N+1 cycles, weight-stationary
      stream B activation vectors        # B + fill cycles
      accumulate column psums            # acc[c][b] += psum (bias on i==0)
    requantize acc -> int8 acts          # or raw INT32 logits (last layer)
```

Layer 0 is 196 input tiles × 8 output blocks = 1568 tile passes; layers 1
and 2 add 32 + 12. Partial sums live in per-column INT32 accumulator banks
*outside* the array between tiles — the array itself stays busy with the
next tile.

## Batching and utilization (the weight-stationary trade-off)

With batch = 1, each loaded tile performs only 16 MACs before being replaced
(~14 cycles per tile ≈ 1.1 MACs/cycle) — weight loading dominates, which is
exactly why weight-stationary designs want data reuse. Streaming B images per
tile amortizes the load: the same weights serve B×16 MACs. At B = 16 the
array sustains ~8× more throughput per image than at B = 1. The Results
section quantifies this and compares against a 1-MAC/cycle sequential
baseline; scaling to 8×8 quadruples MACs/cycle at the same clock.

## Between layers: ping-pong activation banks

Activations are stored 4-way interleaved (element idx → bank idx%4, address
idx/4) so one input tile (4 consecutive elements) reads in a single cycle,
one element from each bank. Two parities of banks alternate: layer l reads
parity l%2 while its requantized outputs write parity (l+1)%2 — output block
j's element j*4+c lands in bank c at address j with no write conflicts.

## Requantization

Between layers, INT32 accumulators are rescaled to INT8 with the integer
multiplier/shift scheme documented in [quantization.md](quantization.md):
`clamp((acc*M + 2^(s-1)) >> s, 0, 127)`, ReLU folded into the low clamp.
The final layer skips this: classification is argmax over raw INT32 logits.

## Design-for-verification hooks

- Inter-PE nets (`a_h`, `psum_v`) and stored weights (`w_dbg`) are
  module-level arrays, runtime-indexable from the testbench — this is what
  makes the complete per-cycle JSON trace (and the visualizer) possible.
- `model/golden.py` defines every arithmetic operation bit-exactly; the
  testbench checks hidden-layer activations and logits against it, not just
  final predictions.
