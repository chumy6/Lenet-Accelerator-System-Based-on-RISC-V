// =======================================
// You need to finish this module
// =======================================

module si_alu #(
    parameter PC_START  = 32'h8000_0000, 
    parameter INST_DW   = 32,
    parameter INST_AW   = 32,
    parameter REG_DW    = 32,
    parameter ALUOP_DW  = 5
)(
    input                   clk,
    input                   rst,
    
    // arithmetic
    input   [ALUOP_DW-1:0]  alu_opcode_i,
    input   [REG_DW-1:0]    operand_1_i,
    input   [REG_DW-1:0]    operand_2_i,
    output  [REG_DW-1:0]    alu_result_o,
    // branch
    input   [INST_AW-1:0]   current_pc_i,
    input                   branch_en_i,
    input   [INST_AW-1:0]   branch_offset_i,
    input                   jump_en_i,
    input   [INST_AW-1:0]   jump_offset_i,
    output                  control_en_o,
    output  [INST_AW-1:0]   control_pc_o
);

    localparam ALU_OP_NOP   = 5'd0 ;
    localparam ALU_OP_ADD   = 5'd1 ;
    localparam ALU_OP_MUL   = 5'd2 ;
    localparam ALU_OP_BNE   = 5'd3 ;
    localparam ALU_OP_JAL   = 5'd4 ;
    localparam ALU_OP_LUI   = 5'd5 ;
    localparam ALU_OP_AUIPC = 5'd6 ;
    localparam ALU_OP_AND   = 5'd7 ;
    localparam ALU_OP_SLL   = 5'd8 ;
    localparam ALU_OP_SLT   = 5'd9 ;
    localparam ALU_OP_BLT   = 5'd10 ;

 // Internal registers
    reg [REG_DW-1:0] alu_result_r;     // Register to hold ALU result
    reg control_en_r;                  // Register to hold control enable
    reg [INST_AW-1:0] control_pc_r;    // Register to hold control PC

    // Assign outputs
    assign alu_result_o = alu_result_r;
    assign control_en_o = control_en_r;
    assign control_pc_o = control_pc_r;

    // ALU Operation
    always @(*) begin
        case (alu_opcode_i)
            5'd1:
            begin
                alu_result_r = operand_1_i + operand_2_i;  // ADD
            end
            5'd2: alu_result_r = operand_1_i * operand_2_i;  // MUL
            5'd7: alu_result_r = operand_1_i & operand_2_i;  // AND
            5'd8: alu_result_r = operand_1_i << operand_2_i[4:0]; // SLL
            5'd9: alu_result_r = ($signed(operand_1_i) < $signed(operand_2_i)) ? 1 : 0; // SLT
            5'd5: alu_result_r = operand_2_i;               // LUI
            5'd6: alu_result_r = current_pc_i + operand_2_i; // AUIPC
            default: alu_result_r = 0;                       // Default: NOP
        endcase
    end

    // Branch and Jump Logic
    always @(*) begin
        control_en_r = 0; // Default: disable control
        control_pc_r = current_pc_i; // Default: no change in PC

        if (branch_en_i) begin
            control_en_r = 1; // Enable branch
            control_pc_r = current_pc_i + branch_offset_i; // Branch target
        end else if (jump_en_i) begin
            control_en_r = 1; // Enable jump
            control_pc_r = current_pc_i + jump_offset_i; // Jump target
        end
    end

endmodule 
