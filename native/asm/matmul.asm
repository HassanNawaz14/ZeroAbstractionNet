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

global matmul_asm_scalar

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

; Mark the stack as non-executable -- without this PE/ELF segment note the
; linker warns "missing .note.GNU-stack section implies executable stack".
section .note.GNU-stack noalloc noexec nowrite progbits