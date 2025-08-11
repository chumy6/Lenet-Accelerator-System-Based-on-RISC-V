; =======================================
; task1_2
; =======================================

# Initialize Base Addresses
lui     x1,    2148532224             ; Base address of A
lui     x2,    2150629376             ; Base address of B
lui     x3,    2152726528             ; Base address of C
lui     x4,    2154823680             ; Base address of D

addi    x9,    x0,     0              ; Row counter (i = 0)
addi    x12,   x0,     8              ; Limit (8 rows/columns)
addi    x22,   x0,     4

outer_loop:
blt     x9,    x12,   next1           ; If i < 8, process row
jal     x0,    end                    ; Else, exit program

next1:
addi    x10,   x0,     0              ; Column counter (j = 0)
addi    x20,   x0,     0              ; Reset accumulator

inner_loop:
blt     x10,   x12,   next2           ; If j < 8, process column
addi    x9,    x9,     1              ; Increment row counter
jal     x0,    outer_loop             ; Go to next row

next2:
addi    x11,   x0,     0              ; Dot product counter (z = 0)
addi    x20,   x0,     0              ; Reset accumulator for dot product

dot_product:
blt     x11,   x12,   next3           ; If z < 8, compute dot product
jal     x0,    bias                   ; Exit to bias addition

next3:
# Load A[i][z]
mul     x14,   x9,    x12             ; 8 * i
add     x14,   x14,   x11             ; 8 * i + z
mul     x14,   x14,   x22               ; (8 * i + z) * 4
add     x15,   x14,   x1              ; Address of A[i][z]
lw      x16,   0(x15)                 ; x16 = A[i][z]

# Load B[z][j]
mul     x14,   x11,   x12             ; 8 * z
add     x14,   x14,   x10             ; 8 * z + j
mul     x14,   x14,   x22               ; (8 * z + j) * 4
add     x15,   x14,   x2              ; Address of B[z][j]
lw      x17,   0(x15)                 ; x17 = B[z][j]

# Multiply and Accumulate
mul     x19,   x16,   x17             ; A[i][z] * B[z][j]
add     x20,   x20,   x19             ; Accumulate result
addi    x11,   x11,   1               ; Increment z counter
jal     x0,    dot_product            ; Repeat dot product

bias:
# Add Bias from C and Store in D
mul     x14,   x9,    x12             ; 8 * i
add     x14,   x14,   x10             ; 8 * i + j
mul     x14,   x14,   x22               ; (8 * i + j) * 4
add     x15,   x14,   x3              ; Address of C[i][j]
lw      x18,   0(x15)                 ; x18 = C[i][j]
add     x20,   x20,   x18             ; Add bias to accumulator

add     x15,   x14,   x4              ; Address of D[i][j]
sw      x20,   0(x15)                 ; D[i][j] = x20

addi    x10,   x10,   1               ; Increment column counter
jal     x0,    inner_loop             ; Repeat inner loop

end:
halt                                   ; End program
