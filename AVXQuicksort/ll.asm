; x64 assembly

.CODE

PUBLIC power	

_TEXT SEGMENT


; size_t simple_bipivot_i32x8_ll(int32_t* dst, int32_t* src, size_t sz, int32_t p);
simple_bipivot_i32x8_ll PROC
	; rcx = dst
	; rdx = src
	; r8  = sz
	; r9  = p

	push rbx	; r 
	push rdi	; l
	push rsi	; sk

	vmovd xmm0, r9d
	vpbroadcastd ymm0, xmm0	; _mm256_set1_epi32

	mov rsi, rdx			; sk = src
	mov rdi, rcx			; l = dst
	lea rbx, [rcx + r8 * 4]	; r = dst+sz
							; load effective instr
							; is a good way to get an 
							; offset pointr like this

iter:
	vmovdqu	ymm1,  YMMWORD PTR [rsi]	; deref sk and load
										; into ymm1 register
	vpcmpgtd ymm1, ymm0, ymm1			; cmp
	vpmovmskb eax, ymm1					; get bitmask
	
	movzx ecx, ax	; 0 pad the mask

	popcnt rdx, rcx	; k = num of bits




	

simple_bipivot_i32x8_ll ENDP




; int testasm(uint64_t base, uint64_t exp);
power PROC
	; rcx = base, rdx = exponent
	mov rax, 1 ; result
power_loop:
	imul rax, rcx
	dec rdx
	jnz power_loop
	ret
power ENDP

_TEXT ENDS
END
