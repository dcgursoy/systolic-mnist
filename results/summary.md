# Results summary (generated)

## Accuracy

| Model | Test set | Accuracy |
|---|---|---|
| float32 PyTorch | 10,000 | 96.63% |
| INT8 golden (Python) | 10,000 | 96.53% |
| RTL simulation | 200 | 98.50% |

RTL vs golden agreement: 100.0% (200/200), logits bit-exact (2000 checked, 0 errors), hidden activations bit-exact (9600 checked, 0 errors).

## Latency (cycles per image)

| Configuration | Cycles/image | vs sequential |
|---|---|---|
| Sequential 1-MAC/cycle baseline | 25,760 | 1.0× |
| 4×4 array, batch 1 | 16,270 | 1.58× |
| 4×4 array, batch 2 | 8,150 | 3.16× |
| 4×4 array, batch 4 | 4,090 | 6.30× |
| 4×4 array, batch 8 | 2,060 | 12.50× |
| 4×4 array, batch 16 | 1,852 | 13.91× |
| 8×8 array, batch 16 | 495 | 52.05× |

## Functional coverage

19/19 bins hit (100%) across regression + stress runs.

## Synthesis (Yosys generic)

| Module | Cells | Logic depth |
|---|---|---|
| PE | 910 | 64 |
| 4×4 array | 8,811 | 77 |
| 8×8 array | 47,807 | 146 |
| full top | 11,303 | 228 |

ECP5-85k place-and-route: Fmax = 65.1 MHz; utilization: TRELLIS_IO 4/365, DP16KD 16/208, MULT18X18D 31/156, TRELLIS_FF 3460/83640, TRELLIS_COMB 5605/83640

