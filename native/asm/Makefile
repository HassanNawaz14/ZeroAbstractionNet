AS = nasm
ASFLAGS = -f elf64
LD = gcc
TARGET = libmatmul_asm.so

# The asm function already speaks the SysV AMD64 ABI natively (registers in
# the right places, callee-saved regs preserved inside the function), so the
# .o links directly into the shared library without a C shim. ctypes looks
# up `matmul_asm_scalar`/`matmul_asm_vectorized`/`matmul_asm` by that exact
# ELF symbol (no leading underscore on Linux).
$(TARGET): matmul.o
	$(LD) -shared -o $(TARGET) matmul.o

matmul.o: matmul.asm
	$(AS) $(ASFLAGS) matmul.asm -o matmul.o

clean:
	rm -f *.o $(TARGET)

.PHONY: clean