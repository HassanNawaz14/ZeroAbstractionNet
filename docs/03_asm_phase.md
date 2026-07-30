# Phase 3 — x86-64 Assembly Matmul Backend

## Prerequisite
Phases 1 and 2 are complete and their Definition of Done checklists are
satisfied. `benchmark_report.md` already has Python and C rows. This
phase adds an `asm` backend following the exact same `ops` interface
pattern as phase 2's `backend_c.py` — read `02_c_phase.md` first if you
haven't, the ctypes marshalling approach here is the same.

## Target ISA — be explicit about this, don't over-reach
Target CPU: Intel 6th-gen Core (Skylake), so:
- **Available**: SSE through SSE4.2, AVX, AVX2, FMA3, BMI1/2.
- **NOT available**: AVX-512 (no 512-bit registers, no `zmm`, no
  AVX-512-only instructions). If any reference material or generated code
  uses `zmm` registers or `vmovaps` with 512-bit operands, that's wrong for
  this target — reject it.
- Vector width for this phase: **256-bit YMM registers**, i.e. 8 `float32`
  or 4 `float64` per instruction.

## Precision decision — switch to float32 here, and why
Phases 1-2 used `double` for exact cross-backend diffing. This phase
switches the asm kernel (and only the asm kernel) to `float32`:
- AVX2 YMM registers hold 8 floats vs. 4 doubles — float32 gets the full
  8-wide SIMD benefit, which is the entire point of hand-writing this in
  assembly.
- Consequence: correctness tests against phases 1/2 can no longer expect
  near-exact agreement (different precision *and* different summation
  order from SIMD lane reduction). Use a **relative tolerance** comparison
  instead of the tight absolute tolerance used in phase 2 — see testing
  section below.
- The training-loop integration (`train.py --backend asm`) will therefore
  run the whole network in float32 when this backend is selected. This is
  a legitimate, expected precision downgrade for this configuration only —
  don't try to silently upcast back to double inside `backend_asm.py`,
  that would defeat the purpose and reintroduce the ctypes overhead this
  phase is trying to eliminate.

## ABI contract (get this exactly right — it's the most common source of
silent corruption bugs in this kind of code)
Target the **System V AMD64 calling convention** (Linux/WSL2 default for
gcc). Function signature, matching phase 2's shape order for consistency:
```c
void matmul_asm(const float *A, const float *B, float *C, int n, int k, int m);
```
Argument-to-register mapping under System V AMD64:
| Arg | Type    | Register  |
|-----|---------|-----------|
| A   | pointer | `rdi`     |
| B   | pointer | `rsi`     |
| C   | pointer | `rdx`     |
| n   | int32   | `rcx` (use `ecx`) |
| k   | int32   | `r8` (use `r8d`)  |
| m   | int32   | `r9` (use `r9d`)  |

Callee-saved registers you'll likely use (`rbx`, `rbp`, `r12`-`r15`) must
be pushed on entry and popped before `ret` if you touch them. YMM
registers `ymm0`-`ymm15` are all caller-saved under SysV — no need to
preserve them, but zero the upper bits with `vzeroupper` before `ret` to
avoid AVX/SSE transition penalties in whatever calls this function next.

## Toolchain
- Assembler: **NASM** (`nasm -f elf64 matmul.asm -o matmul.o`).
- Link into a shared library alongside a tiny C shim, OR link the `.o`
  directly into the `.so` with `gcc`/`ld` — either works since the asm
  function already exports a SysV-compatible symbol; prefer linking the
  object file directly (no C shim needed) since the function already
  speaks the right ABI natively. Verify the exported symbol name matches
  exactly what `ctypes` will look up (`matmul_asm`, no leading underscore
  needed on Linux ELF).

## Directory additions
```
native/asm/
├── matmul.asm
└── Makefile
```

## `native/asm/Makefile`
```makefile
AS = nasm
ASFLAGS = -f elf64
LD = gcc
TARGET = libmatmul_asm.so

$(TARGET): matmul.o
	$(LD) -shared -o $(TARGET) matmul.o

matmul.o: matmul.asm
	$(AS) $(ASFLAGS) matmul.asm -o matmul.o

clean:
	rm -f *.o $(TARGET)

.PHONY: clean
```

## Staged implementation — build and test each stage before starting the
next. Do not jump straight to the final version; each stage is an
independent deliverable that de-risks the next.

### Stage A — scalar, unoptimized asm
Pure scalar `float` triple loop, no SIMD at all — the only goal here is to
validate the ABI plumbing (register reads, memory addressing, the
Python↔ctypes↔asm↔C-library round trip) with the simplest possible
instruction sequence. Export as `matmul_asm_scalar`. This must produce
correct results before you write a single SIMD instruction — if the ABI
wiring is wrong, debugging that is much easier without SIMD complexity on
top.

Addressing pattern for `A[i*k + p]` etc.: compute flat offsets using
`imul`/`lea` and index into the base pointer with `[rdi + rax*4]` (4 bytes
per float32).

### Stage B — AVX2 + FMA vectorized inner loop
Export as `matmul_asm_vectorized`. Vectorize the innermost accumulation
loop 8 floats at a time using `vmovups`/`vfmadd231ps`, with a horizontal
sum at the end of each 8-wide chunk to collapse the YMM accumulator down to
a scalar. Handle the remainder when `k` isn't a multiple of 8 with a
scalar cleanup loop after the vectorized main loop — don't skip this, the
2x2 golden test case (`k=2`) will immediately expose a missing remainder
handler.

Key instructions you'll need: `vmovups` (unaligned load, since we can't
guarantee 32-byte alignment from Python-allocated ctypes buffers),
`vfmadd231ps` (fused multiply-add: `acc = acc + a*b` in one instruction),
and a horizontal-add sequence (`vextractf128` + `vaddps` + `vhaddps`, or
equivalent) to reduce the 8-wide accumulator to one scalar sum.

### Stage C — cache-blocked + vectorized (final version)
Export as `matmul_asm`. Combine stage B's SIMD inner loop with the same
tiling strategy from phase 2's `matmul_blocked` (block size as a constant,
start from the same `BLOCK_SIZE` phase 2 settled on and re-tune if needed —
asm's tighter loop overhead may shift the optimal block size). This is the
version `backend_asm.py` calls by default; keep stages A and B as
permanently-exported symbols too (like phase 2 kept `matmul_naive`) so the
benchmark can show the full progression.

## `ops/backend_asm.py`
Same shape as `backend_c.py` from phase 2, with two differences:
1. Loads `native/asm/libmatmul_asm.so`.
2. Uses `ctypes.c_float` (not `c_double`) for the flatten/unflatten arrays,
   and registers `argtypes` accordingly (`POINTER(c_float)` x3,
   `c_int` x3).
Expose `variant` parameter same as phase 2: `{'scalar', 'vectorized',
'blocked'}` mapping to the three exported symbols, default `'blocked'`.
`add_bias`/`transpose`/`elementwise` are again re-exported from
`backend_python` — same pattern as phase 2 (note: since this backend
operates in float32, double-check these re-exported pure-Python functions
don't silently upcast values back to double in a way that breaks
consistency within a single training run; a simple explicit cast to
`float` — Python's `float` is fine as the in-memory representation between
layers, since the precision loss already happened at the C-boundary
flatten/unflatten step, not in the pure-Python elementwise code).

Update `ops/__init__.py`'s `get_backend("asm")` to return this module.

## Correctness testing — `tests/test_ops_asm.py`
1. Golden 2x2 case again, this time asserting **relative** closeness
   (e.g. `abs(got - expected) / max(1.0, abs(expected)) < 1e-4`) against
   all three variants — scalar should actually match very tightly since
   there's no SIMD reduction reordering; vectorized/blocked need the
   looser tolerance.
2. Property test against `backend_python` (cast its inputs/outputs to
   float32 for a fair comparison) across random shapes, including
   non-multiple-of-8 `k` values specifically (to exercise stage B's
   remainder handling) and non-multiple-of-`BLOCK_SIZE` `n`/`m` values (to
   exercise stage C's block-boundary handling) — these boundary sizes are
   exactly where off-by-one bugs in hand-written asm hide.
3. Full pipeline test: `train.py --backend asm --epochs 500`, same seed as
   phases 1/2, assert final loss is in the same ballpark (this one can't
   use a tight tolerance given the precision change — assert it's within,
   say, 10% of the phase-1/2 final loss, and assert all 4 golden points
   still classify correctly, which is the test that actually matters
   here).

## Benchmarking — final comparison
Extend `benchmark_matmul.py` with `--backend asm --variant
{scalar,vectorized,blocked}`, append to the same `benchmark_results.csv`.
Produce the final version of `benchmark_report.md`: one table, one plot
(size on x-axis, time on log-scale y-axis, one line per
backend/variant combination: python, c-naive, c-blocked, asm-scalar,
asm-vectorized, asm-blocked). This plot is the payoff artifact for the
whole project — it should visually show each stage's improvement clearly
on a log scale.

## Explicit non-goals for this phase (stop here, don't chase further)
- No multithreading (no pthreads, no OpenMP-equivalent hand-rolled
  threading). Single-core only, matching the rest of the project's scope.
- No AVX-512.
- Do not attempt to match or beat system BLAS (OpenBLAS/MKL/whatever
  `numpy.dot` uses under the hood) — that is explicitly out of scope and
  would require register-allocation-level tuning, multi-level blocking,
  and packing strategies far beyond what a finite side project needs.
  Getting within, say, 3-10x of `numpy.dot`'s performance with this
  three-stage approach is already a strong, presentable result.
- Do not add prefetch instructions, alignment-forcing buffer allocators,
  or instruction-scheduling micro-tuning unless stage C's benchmark shows
  a specific, measured bottleneck that clearly points to one of these —
  don't add complexity speculatively.

## Definition of done for phase 3
- [ ] `native/asm/libmatmul_asm.so` builds cleanly from `matmul.asm` via
      `make -C native/asm`.
- [ ] All three stages (scalar, vectorized, blocked) are correct per
      `tests/test_ops_asm.py`, including the non-multiple-of-8 and
      non-multiple-of-`BLOCK_SIZE` boundary cases.
- [ ] `train.py --backend asm` converges and all 4 golden points classify
      correctly.
- [ ] `benchmark_report.md` / plot shows a clear staged progression:
      python < c-naive < c-blocked < asm-scalar < asm-vectorized <
      asm-blocked, with asm-blocked being the fastest but not required to
      beat any external BLAS reference.
- [ ] `animate.py --log-dir logs/<asm-run>` works unchanged.
- [ ] A short `RESULTS.md` at the project root summarizing: max
      network/matrix size trained in a fixed time budget at each phase,
      the final speedup table, and the profiling evidence from
      `profile_baseline.txt` (phase 1) that motivated the whole exercise —
      this is the write-up for your post.
