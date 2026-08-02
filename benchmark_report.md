# Benchmark report - cross-backend comparison

Showcase tier: 2,32,32,1, n=200, 250 epochs, lr 2.5

## Showcase training runs (per epoch)

| backend | epoch (ms) | fwd % | bwd % | upd % | speedup vs python | final loss | golden ok |
|---|---|---|---|---|---|---|---|
| python | 74.0 | 35.2 | 64.6 | 0.2 | 1.0x | 0.016710 | yes |
| c | 20.7 | 36.4 | 62.7 | 0.9 | 3.6x | 0.016710 | yes |

## Square matmul sweep (seconds, best-of-N)

| size | python-naive | c-naive | c-blocked |
|---|---|---|---|
| 16 | 0.000259 | 0.000171 | 0.000177 |
| 32 | 0.002481 | 0.001109 | 0.001159 |
| 64 | 0.0526 | 0.002968 | 0.003822 |
| 128 | 0.2818 | 0.02135 | 0.03079 |
| 256 | 2.637 | 0.1551 | 0.09788 |
| 512 | 26.11 | 1.326 | 0.5725 |
| 1024 | - | 16.37 | 3.112 |
| 2048 | - | - | 18.52 |

## Shaped matmul (showcase shapes, seconds)

| shape | python | c |
|---|---|---|
| 200x1x32 | 0.0015597 | 0.00021664 |
| 200x2x32 | 0.0031432 | 0.00023876 |
| 200x32x1 | 0.00046882 | 0.00054176 |
| 200x32x2 | 0.00085007 | 0.00060471 |
| 200x32x32 | 0.016261 | 0.0010355 |
