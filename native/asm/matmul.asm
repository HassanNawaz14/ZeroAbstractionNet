; matmul.asm -- x86-64 assembly matmul kernels (phase 3).
;
; System V AMD64 calling convention (Linux/WSL2 default of gcc).
; Contract (matching the C/ctypes shape order from phase 2):
;   void matmul_asm_scalar(const float *A, const float *B, float *C,
;                          int n, int k, int m);
; Argument registers: A=rdi B=rsi C=rdx n=ecx k=r8d m=r9d.
;
; Callee-saved registers rbx/rbp/r12-r15 are pushed on entry and popped
; before ret if this function touches them. YMM/XMM registers are
; caller-saved under SysV; vzeroupper is issued before ret to avoid
; AVX/SSE transition penalties in whatever calls us next.
;
; Target ISA: Intel 6th-gen Core (Skylake) -- AVX2 + FMA3 available,
; NO AVX-512 (no zmm). Vector width is 256-bit YMM = 8 x float32.
; 256-bit YMM register 8x float32 per instruction.

section .text

; Cache block size for matmul_asm (phase-3 stage C). Rows are tiled in both
; the output-row dimension and the k dimension (phase-2 strategy) with the
; stage-B FMA inner loop running inside each tile; the 8-wide output-column
; groups segment the m dimension (columns are always processed 8 at a time).
; Tuned from phase 2's BLOCK_SIZE=96 (float32 tiles are half the bytes, so
; the asm optimum may differ -- re-tune if stage C's sweep shows a shift).
%define BLOCK_SIZE 96

global matmul_asm_scalar
global matmul_asm_vectorized
global matmul_asm

; ------------------------------------------------------------------
; Stage A -- pure scalar float32 triple loop, NO SIMD.
; Sole purpose: prove the Python <-> ctypes <-> C-dlopen <-> asm ABI
; plumbing (register reads, memory addressing, the round trip) with the
; simplest possible instruction sequence, BEFORE any SIMD is written.
; This stays in the codebase forever as the benchmark baseline stage.
; Deliberately NOT optimized -- that is the point of the stage A/B/C
; progression.
; ------------------------------------------------------------------
matmul_asm_scalar:
    push   rbx
    push   rbp
    push   r12
    push   r13
    push   r14
    push   r15

    mov    r12, rdi            ; A
    mov    r13, rsi            ; B
    mov    r14, rdx            ; C
    movsxd r15, ecx            ; n
    movsxd rbx, r8d            ; k
    movsxd rbp, r9d            ; m

    ; ---- zero C (n*m floats): we accumulate with += below ----
    mov    rax, r15
    imul   rax, rbp
    lea    r10, [r14 + rax*4]  ; one-past-end of C
    xorps  xmm0, xmm0
    mov    r11, r14
.zero_loop:
    cmp    r11, r10
    jae    .zero_done
    movss  [r11], xmm0
    add    r11, 4
    jmp    .zero_loop
.zero_done:

    ; ---- C[i*m+j] += A[i*k+p] * B[p*m+j] ----
    xor    rcx, rcx            ; i
.loop_i:
    xor    r8, r8              ; j
.loop_j:
    xorps  xmm0, xmm0          ; sum accumulator
    xor    r9, r9              ; p
.loop_p:
    mov    rax, rcx            ; flat A index = i*k + p
    imul   rax, rbx
    add    rax, r9
    movss  xmm1, [r12 + rax*4]
    mov    rax, r9             ; flat B index = p*m + j
    imul   rax, rbp
    add    rax, r8
    mulss  xmm1, [r13 + rax*4]
    addss  xmm0, xmm1
    inc    r9
    cmp    r9, rbx
    jb     .loop_p
    mov    rax, rcx            ; flat C index = i*m + j
    imul   rax, rbp
    add    rax, r8
    movss  [r14 + rax*4], xmm0
    inc    r8
    cmp    r8, rbp
    jb     .loop_j
    inc    rcx
    cmp    rcx, r15
    jb     .loop_i

    vzeroupper
    pop    r15
    pop    r14
    pop    r13
    pop    r12
    pop    rbp
    pop    rbx
    ret

; ------------------------------------------------------------------
; Stage B -- AVX2+FMA vectorized inner loop over OUTPUT lanes.
;
; For each output row i we accumulate 8 output columns at once:
;   out[j..j+7] = sum_p A[i*k+p] * B[p*m + j..j+7]
; so the FMA inner loop is 8-wide in the OUTPUT dimension:
;   vbroadcastss (a scalar A value) + vmovups (8 contiguous B floats,
;   since B rows are contiguous) + vfmadd231ps (acc += a*b).
; Writing the accumulated ymm straight to C removes the need for a
; horizontal reduce entirely -- the "k-direction vectorize + hsum" shape
; from the doc is one valid split; output-lane vectorization is the other
; and is what hand-written BLAS-style kernels do (contiguous B loads).
;
; Remainder handling: when fewer than 8 output columns remain (m % 8 != 0,
; e.g. the golden k=2 -> m=2 case) we fall back to a scalar loop for the
; tail columns. k never needs unrolling, so any k (1, 2, 7, 200...) is fine.
matmul_asm_vectorized:
    push   rbx
    push   rbp
    push   r12
    push   r13
    push   r14
    push   r15

    mov    r12, rdi            ; A
    mov    r13, rsi            ; B
    mov    r14, rdx            ; C
    movsxd r15, ecx            ; n
    movsxd rbx, r8d            ; k
    movsxd rbp, r9d            ; m

    ; ---- zero C (we only write, never re-read, but keep it deterministic) ----
    mov    rax, r15
    imul   rax, rbp
    lea    r10, [r14 + rax*4]
    xorps  xmm0, xmm0
    mov    r11, r14
.zero_loop:
    cmp    r11, r10
    jae    .zero_done
    movss  [r11], xmm0
    add    r11, 4
    jmp    .zero_loop
.zero_done:

    ; ---- i over rows ----
    xor    rcx, rcx
.vloop_i:
    xor    r8, r8              ; jj (vector column group)
.vcols:
    mov    rax, rbp
    sub    rax, r8
    cmp    rax, 8
    jb     .vtail
    ; ---- 8-wide accumulate for columns jj..jj+7 ----
    vpxor  ymm0, ymm0, ymm0    ; acc
    xor    r9, r9              ; p
.vec_loop:
    mov    rax, rcx            ; A[i*k + p]
    imul   rax, rbx
    add    rax, r9
    vbroadcastss ymm1, [r12 + rax*4]
    mov    rax, r9             ; B[p*m + jj .. +7]
    imul   rax, rbp
    add    rax, r8
    vmovups ymm2, [r13 + rax*4]
    vfmadd231ps ymm0, ymm1, ymm2 ; acc = acc + A_broadcast * B_band
    inc    r9
    cmp    r9, rbx
    jb     .vec_loop
    mov    rax, rcx            ; C[i*m + jj .. +7]
    imul   rax, rbp
    add    rax, r8
    vmovups [r14 + rax*4], ymm0
    add     r8, 8
    jmp    .vcols

    ; ---- scalar tail for the last m%8 output columns ----
.vtail:
    cmp    r8, rbp
    jae    .vrow_done
.vtail_col:
    xorps  xmm0, xmm0          ; sum (xmm0 is the low half of ymm0, fine)
    xor    r9, r9
.vtail_p:
    mov    rax, rcx             ; A[i*k + p]
    imul   rax, rbx
    add    rax, r9
    movss  xmm1, [r12 + rax*4]
    mov    rax, r9              ; B[p*m + j]
    imul   rax, rbp
    add    rax, r8
    mulss  xmm1, [r13 + rax*4]
    addss  xmm0, xmm1
    inc    r9
    cmp    r9, rbx
    jb     .vtail_p
    mov    rax, rcx             ; C[i*m + j]
    imul   rax, rbp
    add    rax, r8
    movss  [r14 + rax*4], xmm0
    inc    r8
    cmp    r8, rbp
    jb     .vtail
.vrow_done:
    inc    rcx
    cmp    rcx, r15
    jb     .vloop_i

    vzeroupper
    pop    r15
    pop    r14
    pop    r13
    pop    r12
    pop    rbp
    pop    rbx
    ret

; ------------------------------------------------------------------
; Stage C -- cache-blocked + vectorized (final default kernel).
;
; Tiling strategy from phase 2's matmul_blocked, SIMD body from stage B:
;   outer  : output-row blocks (ii) and k-blocks (kk)
;   middle : 8-wide output-column groups (j8)
;   inner  : row i x p in [kk..k_end) FMA accumulation
; C accumulates with += across k-blocks; the scalar tail handles the last
; m%8 columns (this also covers m < 8 entirely, e.g. the golden k=2 -> m=2
; case). BLOCK_SIZE is a %define above so it can be re-tuned.
; Register map (after prologue):
;   rsi=BLOCK_SIZE  r12=A  r13=B  r14=C  r15=n  rbx=k  rbp=m
;   rcx=ii  r8=i_end  r9=k_end  r10=kk  r11=j8  rdi=i  rdx=p
matmul_asm:
    push   rbx
    push   rbp
    push   r12
    push   r13
    push   r14
    push   r15

    mov    r12, rdi            ; A
    mov    r13, rsi            ; B
    mov    r14, rdx            ; C
    movsxd r15, ecx            ; n
    movsxd rbx, r8d            ; k
    movsxd rbp, r9d            ; m
    mov    rsi, BLOCK_SIZE

    ; ---- zero C ----
    mov    rax, r15
    imul   rax, rbp
    lea    r10, [r14 + rax*4]
    xorps  xmm0, xmm0
    mov    r11, r14
.zero_loop:
    cmp    r11, r10
    jae    .zero_done
    movss  [r11], xmm0
    add    r11, 4
    jmp    .zero_loop
.zero_done:

    ; ---- outer row-block loop (ii) ----
    xor    rcx, rcx
.iblock:
    cmp    rcx, r15
    jae    .b_done
    mov    r8, rcx
    add    r8, rsi            ; i_end = ii + BLOCK_SIZE (capped at n)
    cmp    r8, r15
    cmova  r8, r15

    ; ---- outer k-block loop (kk) ----
    xor    r10, r10
.kblock:
    cmp    r10, rbx
    jae    .iblock_next
    mov    r9, r10
    add    r9, rsi            ; k_end = kk + BLOCK_SIZE (capped at k)
    cmp    r9, rbx
    cmova  r9, rbx

    ; ---- 8-wide output-column groups ----
    xor    r11, r11
.j8:
    lea    rax, [r11 + 8]
    cmp    rax, rbp
    ja     .jtail
    mov    rdi, rcx           ; i = ii
.irow:
    cmp    rdi, r8            ; i >= i_end?
    jge    .j8_next
    vpxor  ymm0, ymm0, ymm0   ; acc lanes for outputs j8..j8+7
    mov    rdx, r10           ; p = kk
.psum:
    cmp    rdx, r9            ; p >= k_end?
    jge    .store8
    mov    rax, rdi
    imul   rax, rbx
    add    rax, rdx
    vbroadcastss ymm1, [r12 + rax*4]   ; A[i*k + p] -> all 8 lanes
    mov    rax, rdx
    imul   rax, rbp
    add    rax, r11
    vmovups ymm2, [r13 + rax*4]        ; B[p*m + j8 .. +7]
    vfmadd231ps ymm0, ymm1, ymm2
    inc    rdx
    jmp    .psum
.store8:
    mov    rax, rdi
    imul   rax, rbp
    add    rax, r11
    vmovups ymm3, [r14 + rax*4]        ; existing C values (previous k-blocks)
    vaddps ymm0, ymm0, ymm3
    vmovups [r14 + rax*4], ymm0
    inc    rdi
    jmp    .irow
.j8_next:
    add    r11, 8
    jmp    .j8

    ; ---- scalar tail: columns j8..m-1 (all rows, current k-block) ----
.jtail:
    cmp    r11, rbp
    jae    .kblock_next
.tcol:
    mov    rdi, rcx           ; i = ii
.trow:
    cmp    rdi, r8            ; i >= i_end?
    jge    .tcol_next
    xorps  xmm0, xmm0
    mov    rdx, r10           ; p = kk
.tp:
    cmp    rdx, r9            ; p >= k_end?
    jge    .tsum
    mov    rax, rdi
    imul   rax, rbx
    add    rax, rdx
    movss  xmm1, [r12 + rax*4]   ; A[i*k + p]
    mov    rax, rdx
    imul   rax, rbp
    add    rax, r11
    mulss  xmm1, [r13 + rax*4]   ; B[p*m + j]
    addss  xmm0, xmm1
    inc    rdx
    jmp    .tp
.tsum:
    mov    rax, rdi
    imul   rax, rbp
    add    rax, r11
    movss  xmm1, [r14 + rax*4]   ; existing C value
    addss  xmm0, xmm1
    movss  [r14 + rax*4], xmm0
    inc    rdi
    jmp    .trow
.tcol_next:
    inc    r11
    jmp    .jtail

.kblock_next:
    add    r10, rsi
    jmp    .kblock
.iblock_next:
    add    rcx, rsi
    jmp    .iblock
.b_done:
    vzeroupper
    pop    r15
    pop    r14
    pop    r13
    pop    r12
    pop    rbp
    pop    rbx
    ret

; Mark the stack as non-executable -- without this PE/ELF segment note the
; linker warns "missing .note.GNU-stack section implies executable stack".
section .note.GNU-stack noalloc noexec nowrite progbits