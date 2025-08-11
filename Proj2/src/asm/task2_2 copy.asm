###############################################################################
# Step 1: Base Address Setup (A=0x80100000, B=0x80300000, C=0x80500000, D=0x80700000)
###############################################################################
lui   x10, 0x80100    ; vadd.vi vx0, vx0, 0         # x10 = 0x80100000 (base of A)
lui   x11, 0x80300    ; vadd.vi vx0, vx0, 0         # x11 = 0x80300000 (base of B)
lui   x12, 0x80500    ; vadd.vi vx0, vx0, 0         # x12 = 0x80500000 (base of C)
lui   x13, 0x80700    ; vadd.vi vx0, vx0, 0         # x13 = 0x80700000 (base of D)

###############################################################################
# Step 2: Loop Setup
#   x14 = row index (0..7)
#   x15 = col index (0..7)
###############################################################################
addi  x14, x0, 0      ; vadd.vi vx0, vx0, 0         # row = 0
loop_rows:
slti  x16, x14, 8     ; vadd.vi vx0, vx0, 0         # check row < 8?
blt   x16, x0, end_rows ; vadd.vi vx0, vx0, 0       # if x16>=0 => row>=8 => done
addi  x15, x0, 0      ; vadd.vi vx0, vx0, 0         # col = 0

loop_cols:
slti  x17, x15, 8     ; vadd.vi vx0, vx0, 0         # check col < 8?
blt   x17, x0, done_cols ; vadd.vi vx0, vx0, 0      # if col>=8 => done_cols

###############################################################################
# Step 3: Load Row of A and Row of B
#   We'll load row "x14" of A into vx2, row "x14" of B into vx3
###############################################################################
sll   x19, x14, 5     ; vadd.vi vx0, vx0, 0         # x19 = row << 5 => row*32
add   x19, x10, x19   ; vadd.vi vx0, vx0, 0         # x19 = baseA + offset
addi  x0,  x0,  0     ; vle32.v vx2, (x19)          # load row of A into vx2

sll   x20, x14, 5     ; vadd.vi vx0, vx0, 0         # x20 = row << 5 => row*32
add   x20, x11, x20   ; vadd.vi vx0, vx0, 0         # x20 = baseB + offset
addi  x0,  x0,  0     ; vle32.v vx3, (x20)          # load row of B into vx3

###############################################################################
# Step 4: Multiply + Accumulate
#   - Zero vx6
#   - Multiply vx2 * vx3 => vx4
#   - Accumulate vx4 into vx6
###############################################################################
add   x5, x5, x0      ; vadd.vi vx6, vx0, 0         # vx6 = 0
addi  x0, x0, 0       ; vmul.vv vx4, vx2, vx3       # vx4 = vx2 * vx3
addi  x0, x0, 0       ; vadd.vv vx6, vx6, vx4       # vx6 += vx4

###############################################################################
# Step 5: Add Bias from C
#   Load row "x14" of C into vx7, then vadd.vv vx6, vx6, vx7
###############################################################################
sll   x21, x14, 5     ; vadd.vi vx0, vx0, 0         # x21 = row << 5 => row*32
add   x21, x12, x21   ; vadd.vi vx0, vx0, 0         # x21 = baseC + offset
addi  x0,  x0,  0     ; vle32.v vx7, (x21)          # load row of C into vx7
addi  x0,  x0,  0     ; vadd.vv vx6, vx6, vx7       # vx6 = vx6 + vx7

###############################################################################
# Step 6: Store the Result into D
#   store row "x14" result from vx6
###############################################################################
sll   x21, x14, 5     ; vadd.vi vx0, vx0, 0         # x21 = row << 5 => row*32
add   x21, x13, x21   ; vadd.vi vx0, vx0, 0         # x21 = baseD + offset
addi  x0,  x0,  0     ; vse32.v vx6, (x21)          # store final result to D

###############################################################################
# Increment col, jump back
###############################################################################
addi  x15, x15, 1     ; vadd.vi vx0, vx0, 0         # col++
jal   x0,   loop_cols ; vadd.vi vx0, vx0, 0

done_cols:
###############################################################################
# Increment row, jump back
###############################################################################
addi  x14, x14, 1     ; vadd.vi vx0, vx0, 0         # row++
jal   x0,   loop_rows ; vadd.vi vx0, vx0, 0

end_rows:
###############################################################################
# Step 7: Finalize - jump to "finish" or infinite loop
###############################################################################
add   x0, x0, x0      ; vadd.vi vx0, vx0, 0         # no-op pair
jal   x0, finish      ; vadd.vi vx0, vx0, 0         # jump

finish:
addi  x0, x0, 0       ; vadd.vi vx0, vx0, 0         # no-op
jal   x0, finish      ; vadd.vi vx0, vx0, 0         # infinite loop
