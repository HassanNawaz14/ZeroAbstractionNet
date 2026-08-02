CC = gcc
# -O2 only, deliberately NOT -O3 / -march=native / -mavx2 / -ffast-math:
# phase 2's numbers must mean "what a careful but portable C
# implementation gets you", so that phase 3's hand-written SIMD has a
# fair, distinct story to tell ("here's what happens when we go further
# than the compiler will by default"). gcc -O3 -march=native would
# auto-vectorize this loop into something close to phase 3's kernel — that
# is exactly what we're holding back.
#
# -ffp-contract=off additionally disables FMA contraction (a*b+c fused into
# ONE vfmadd with a single rounding). Python's scalar loop always rounds
# twice, and this net trains at lr 2.5 — a chaotic, amplifying regime where
# a 1-ulp difference cascades into visibly different trajectories (and a
# different final minimum). We need C's output to be BIT-IDENTICAL to the
# pure-python backend at demo scale so cross-backend logs/animation match.
# Phase 3's assembly introduces FMA deliberately; that's the phase for it.
CFLAGS = -O2 -fPIC -Wall -Wextra -ffp-contract=off
TARGET = libmatmul.so

$(TARGET): matmul.c matmul.h
	$(CC) $(CFLAGS) -shared -o $(TARGET) matmul.c

clean:
	rm -f $(TARGET)

.PHONY: clean
