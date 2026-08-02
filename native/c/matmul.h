#ifndef MATMUL_H
#define MATMUL_H

// A is n x k (row-major), B is k x m (row-major), C is n x m (row-major,
// pre-allocated by the caller, this function only writes into it).
void matmul_naive(const double *A, const double *B, double *C, int n, int k, int m);

// Same contract, but with loop order and access pattern optimized for
// cache locality (see matmul.c for details). Both functions must be
// exported — the naive one stays as a benchmarking baseline forever, it's
// not dead code to delete once the optimized one exists.
void matmul_blocked(const double *A, const double *B, double *C, int n, int k, int m);

#endif
