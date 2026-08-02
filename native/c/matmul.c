#include "matmul.h"

// Cache-block size for matmul_blocked. A BLOCK_SIZE x BLOCK_SIZE tile of
// doubles is BLOCK_SIZE**2 * 8 bytes; the i-k-j inner loops touch one A, one B
// and one C tile at a time, so big tiles must stay in cache.
// Tuned empirically on this machine (i5-6200U, 32 KB L1 / 256 KB L2 per core):
// BS=16 2.88s, 32 3.04s, 48 2.79s, 64 2.60s, 96 2.48s, 128 2.53s at n=1024.
// 96 was fastest; expose the #define so it can be re-tuned on other boxes.
#define BLOCK_SIZE 96

void matmul_naive(const double *A, const double *B, double *C, int n, int k, int m) {
    // Direct triple loop, deliberately NOT optimized: B[p * m + j] strides
    // through memory by m doubles every inner-loop iteration — cache-hostile
    // by design. This baseline isolates "speedup from compiled C vs
    // interpreted Python" from "speedup from smarter access patterns", so it
    // must never accidentally become optimized.
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * m + j];
            }
            C[i * m + j] = sum;
        }
    }
}

void matmul_blocked(const double *A, const double *B, double *C, int n, int k, int m) {
    // Zero-initialize C first: the i-k-j loops below accumulate with +=.
    for (int i = 0; i < n * m; i++) {
        C[i] = 0.0;
    }

    // Three nested "outer" loops over block indices, with the i-k-j loops
    // from the doc running inside each tile. Two independent improvements:
    //
    // 1. Loop reorder (i-k-j): the innermost loop walks contiguous memory in
    //    both B and C, instead of naive's strided B access.
    // 2. Cache tiling: process fixed BLOCK_SIZE x BLOCK_SIZE tiles so the
    //    working set stays resident in L1/L2 for large n.
    for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
        for (int pp = 0; pp < k; pp += BLOCK_SIZE) {
            for (int jj = 0; jj < m; jj += BLOCK_SIZE) {
                for (int i = ii; i < ii + BLOCK_SIZE && i < n; i++) {
                    for (int p = pp; p < pp + BLOCK_SIZE && p < k; p++) {
                        double a_ip = A[i * k + p];
                        for (int j = jj; j < jj + BLOCK_SIZE && j < m; j++) {
                            C[i * m + j] += a_ip * B[p * m + j];
                        }
                    }
                }
            }
        }
    }
}
