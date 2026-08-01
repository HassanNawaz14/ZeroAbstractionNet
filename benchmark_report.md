# Benchmark report - cross-backend comparison

Showcase tier: 2,32,32,1, n=200, 250 epochs, lr 2.5

## Showcase training runs (per epoch)

| backend | epoch (ms) | fwd % | bwd % | upd % | speedup vs python | final loss | golden ok |
|---|---|---|---|---|---|---|---|
| python | 264.5 | 34.8 | 65.0 | 0.2 | 1.0x | 0.016710 | yes |

## Square matmul sweep (seconds, best-of-N)

| size | python-naive |
|---|---|
| 16 | 0.000421 |
| 32 | 0.003027 |
| 64 | 0.05063 |
| 128 | 0.5144 |
| 256 | 4.326 |
| 512 | 40.54 |

## Shaped matmul (showcase shapes, seconds)

| shape | python |
|---|---|
| 200x1x32 | 0.0050953 |
| 200x2x32 | 0.0047925 |
| 200x32x1 | 0.0012189 |
| 200x32x2 | 0.0024778 |
| 200x32x32 | 0.065008 |
