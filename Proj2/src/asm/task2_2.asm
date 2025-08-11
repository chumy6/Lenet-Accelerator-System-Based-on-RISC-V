; =======================================
; task2_2
; =======================================

lui     x1,    2149580800             ; nop                                       ; Base address of A.T
lui     x2,    2151677952             ; nop                                       ; Base address of B.T
lui     x3,    2153775104             ; nop                                       ; Base address of C.T
lui     x4,    2155872256             ; nop                                       ; Base address of D.T

lui     x5,    2148532224             ; nop                                       ; Base address of A
lui     x6,    2150629376             ; nop                                       ; Base address of B
lui     x7,    2152726528             ; nop                                       ; Base address of C
lui     x8,    2154823680             ; nop                                       ; Base address of D

addi    x19, x2, 0                    ; nop                                       ; x19 walks B.T
addi    x18, x7, 0                    ; nop                                       ; x18 walks through C
addi    x17, x8, 0                    ; nop                                       ; x17 walks through D
addi    x9,    x0,     0              ; nop                                       ; Row counter (i = 0)
addi    x12,   x0,     8              ; nop                                       ; Limit (8 rows/columns)
addi    x22,   x0,     4              ; nop                                       ;

outer_loop:
blt     x9,    x12,   next1           ; nop                                       ; If i < 8, process row
jal     x0,    end                    ; nop                                       ; Else, exit program

next1:
addi    x10,   x0,     0              ; nop                                       ; Column counter (j = 0)
addi    x19, x2, 0                    ; nop                                       ; rest x19 to B.T

inner_loop:
blt     x10,   x12,   dot_product     ; nop                                       ; If j < 8, process column
addi    x9,    x9,     1              ; nop                                       ; Increment row counter
addi    x5, x5, 32                  ; nop                                       ; next row of A
jal     x0,    outer_loop             ; nop                                       ; Go to next row

dot_product:
nop                                     ; vle32.v vx0, x5, 1                      ; load A
nop                                     ; vle32.v vx1, x19, 1                      ; load B.T
nop                                     ; vmul.vv vx2, vx1, vx0, 1                ;
nop                                     ; vse32.v vx2, x3, 1                      ; save vector reg to not-used address and do accumulate

addi    x13, x0, 0                    ; nop                                       ; x13 as counter
addi    x14, x3, 0                    ; nop                                       ; x14 as memory index
addi    x15, x0, 0                    ; nop                                       ; x15 as accumulator

accmulate:
blt     x13, x12,  next2              ; nop                                       ;
jal     x0, save                      ; nop                                       ;

next2:
lw      x16, 0(x14)                  ; nop                                       ;
add     x15, x15, x16                ; nop                                       ;
addi    x13, x13, 1                  ; nop                                       ;
addi    x14, x14, 4                  ; nop                                       ;
jal     x0, accmulate                ; nop                                       ;

save:
lw      x25, 0(x18)                  ; nop                                       ; C
add     x25, x25, x15                ; nop                                       ; D = A * B + C
sw      x25, 0(x17)                   ; nop                                       ; save to D
addi    x17, x17, 4                  ; nop                                       ; update D
addi    x18, x18, 4                  ; nop                                       ; update C

addi    x10,   x10,   1              ; nop                                       ; Increment column counter
addi    x19, x19, 32                ; nop                                       ; next column for B
jal     x0,    inner_loop            ; nop                                       ; Repeat inner loop

end:
halt                                   ; nop                                       ; End program
