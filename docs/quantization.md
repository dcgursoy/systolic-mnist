# Quantization scheme

The accelerator runs the entire MLP in integer arithmetic. The scheme is
symmetric per-tensor INT8 with INT32 accumulation and integer requantization —
the same approach used by TFLite / gemmlowp and most production edge
accelerators.

## Number formats

| Signal | Format | Range |
|---|---|---|
| Input pixels | int8 (unsigned in practice) | 0 … 127 |
| Activations (post-ReLU) | int8 | 0 … 127 |
| Weights | int8, symmetric | −127 … 127 |
| Biases | int32, scale = s_in · s_w | |
| Accumulators | int32 | worst case 784·127·127 ≈ 1.26e7 ≪ 2³¹ |
| Requant multiplier M | int16 (normalized to [2¹⁴, 2¹⁵)) | per layer |

## Mapping floats to integers

For each tensor, a real *scale* s maps integers to reals: `real ≈ s · q`.

- Input: pixels in [0,1] are mapped with `q = round(x·127)`, so `s_in = 1/127`.
- Weights: `s_w = max|W| / 127`, `W_q = round(W / s_w)` (per layer).
- Bias is added inside the accumulator, so it uses the accumulator's scale:
  `b_q = round(b / (s_in · s_w))` as int32.

## Requantization (between layers)

The accumulator of layer *l* has scale `s_in·s_w`; the next layer expects an
int8 activation with scale `s_out` (calibrated as max-abs of the float
activation over 2000 training images / 127). The real rescale factor

    m = s_in · s_w / s_out          (0 < m < 1)

is represented as an integer multiplier and shift, `m ≈ M / 2^shift`, with M
normalized into [2¹⁴, 2¹⁵) so the multiply keeps ~15 bits of precision:

    y = clamp( (acc · M + 2^(shift−1)) >> shift , 0, 127 )

The `>>` is an arithmetic shift; the additive term implements round-half-up.
Clamping the low side at 0 folds ReLU into the requantizer for free.

The final layer skips requantization: `argmax` over the raw int32
accumulators gives the classification directly, and the float logits (for
the confidence display) are recovered as `acc · s_in · s_w`.

## Bit-exactness contract

`model/golden.py` is the executable specification of this arithmetic. The
RTL must match it bit-for-bit: same accumulation order per output (biases
preloaded, weights·activations accumulated), same rounding, same clamps.
The testbench checks intermediate accumulator values, not just final
predictions, against this model.
