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
| 4×4 array, batch 1 | 24,211 | 1.06× |
| 4×4 array, batch 2 | 12,919 | 1.99× |
| 4×4 array, batch 4 | 7,273 | 3.54× |
| 4×4 array, batch 8 | 4,450 | 5.79× |
| 4×4 array, batch 16 | 3,095 | 8.32× |
| 8×8 array, batch 16 | 1,095 | 23.52× |

## Functional coverage

19/19 bins hit (100%) across regression + stress runs.

## Synthesis (Yosys generic)

| Module | Cells | Logic depth |
|---|---|---|
| PE | 901 | 63 |
| 4×4 array | 8,682 | 75 |
| 8×8 array | 47,316 | 144 |
| full top | 10,858 | 166 |

ECP5-85k place-and-route: Fmax = 43.2 MHz; utilization: TRELLIS_IO 4/365, DP16KD 16/208, MULT18X18D 30/156, TRELLIS_FF 3109/83640, TRELLIS_COMB 5412/83640

