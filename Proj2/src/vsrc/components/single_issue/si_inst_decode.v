// =======================================
// You need to finish this module
// =======================================


`include "define_rv32im.v"

module si_inst_decode #(
    parameter INST_DW   = 32,
    parameter INST_AW   = 32,
    parameter MEM_AW    = 32,
    parameter REG_DW    = 32,
    parameter REG_AW    = 5,
    parameter ALUOP_DW  = 5

) (
    input                   clk,
    input                   rst,
    // instruction
    input   [INST_DW-1:0]   inst_i,
    // regfile
    output                  rs1_en_o,
    output  [REG_AW-1:0]    rs1_addr_o,
    input   [REG_DW-1:0]    rs1_dout_i,
    output                  rs2_en_o,
    output  [REG_AW-1:0]    rs2_addr_o,
    input   [REG_DW-1:0]    rs2_dout_i,
    // alu
    output  [ALUOP_DW-1:0]  alu_opcode_o,
    output  [REG_DW-1:0]    operand_1_o,
    output  [REG_DW-1:0]    operand_2_o,
    output                  branch_en_o,
    output  [INST_AW-1:0]   branch_offset_o,
    output                  jump_en_o,
    output  [INST_AW-1:0]   jump_offset_o,
    // mem-access
    output                  mem_ren_o,
    output                  mem_wen_o,
    output  [INST_DW-1:0]   mem_din_o,
    // write-back
    output                  id_wb_en_o,
    output                  id_wb_sel_o,
    output  [REG_AW-1:0]    id_wb_addr_o 
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

 // Extract fields from instruction
    wire [6:0] opcode  = inst_i[6:0];
    wire [2:0] funct3  = inst_i[14:12];
    wire [6:0] funct7  = inst_i[31:25];
    wire [4:0] rs1     = inst_i[19:15];
    wire [4:0] rs2     = inst_i[24:20];
    wire [4:0] rd      = inst_i[11:7];
    wire [11:0] imm_i  = inst_i[31:20]; // Immediate for I-type
    wire [11:0] imm_s  = {inst_i[31:25], inst_i[11:7]}; // Immediate for S-type
    wire [12:0] imm_b  = {inst_i[31], inst_i[7], inst_i[30:25], inst_i[11:8], 1'b0}; // Immediate for B-type
    wire [31:0] imm_u  = {inst_i[31:12], 12'b0}; // Immediate for U-type
    wire [20:0] imm_j  = {inst_i[31], inst_i[19:12], inst_i[20], inst_i[30:21], 1'b0}; // Immediate for J-type

    // Default values
    assign rs1_en_o        = (opcode == `OPCODE_ADD || opcode == `OPCODE_ADDI ||
                               opcode == `OPCODE_LW || opcode == `OPCODE_SW ||
                               opcode == `OPCODE_BNE || opcode == `OPCODE_BLT ||
                               opcode == `OPCODE_JALR || opcode == `OPCODE_MUL ||
                               opcode == `OPCODE_SLL || opcode == `OPCODE_SLTI ||
                               opcode == `OPCODE_AND);
    assign rs2_en_o        = (opcode == `OPCODE_ADD || opcode == `OPCODE_SW ||
                               opcode == `OPCODE_BNE || opcode == `OPCODE_BLT ||
                               opcode == `OPCODE_MUL || opcode == `OPCODE_SLL ||
                               opcode == `OPCODE_AND);
    assign rs1_addr_o      = rs1;
    assign rs2_addr_o      = rs2;

    // Temporary reg variable for ALU opcode
    reg [ALUOP_DW-1:0] alu_opcode_temp;

    // Connect the temp variable to the output wire
    assign alu_opcode_o = alu_opcode_temp;

    // Logic to determine ALU operation
    always @(*) begin
        // Default assignment
        alu_opcode_temp = ALU_OP_NOP;

        case (opcode)
            // R-type Instructions (e.g., ADD, AND, SLL, MUL)
            `OPCODE_ADD: begin
                if (funct3 == `FUNCT3_ADD && funct7 == `FUNCT7_ADD) begin
                    alu_opcode_temp = ALU_OP_ADD; // ADD
                end else if (funct3 == `FUNCT3_AND && funct7 == `FUCNT7_AND) begin
                    alu_opcode_temp = ALU_OP_AND; // AND
                end else if (funct3 == `FUNCT3_SLL && funct7 == `FUCNT7_SLL) begin
                    alu_opcode_temp = ALU_OP_SLL; // SLL
                end else if (funct3 == `FUNCT3_MUL && funct7 == `FUNCT7_MUL) begin
                    alu_opcode_temp = ALU_OP_MUL; // MUL
                end
            end

            // I-type Instructions (e.g., ADDI, SLTI, LW)
            `OPCODE_ADDI: begin
                if (funct3 == `FUNCT3_ADDI) begin
                    alu_opcode_temp = ALU_OP_ADD; // ADDI
                end
                if (funct3 == `FUNCT3_SLTI) begin
                    alu_opcode_temp = ALU_OP_SLT; // SLTI
                end
            end
            `OPCODE_LW: begin
                if (funct3 == `FUNCT3_LW) begin
                    alu_opcode_temp = ALU_OP_ADD; // LW (Address Calculation)
                end
            end

            // S-type Instructions (e.g., SW)
            `OPCODE_SW: begin
                if (funct3 == `FUNCT3_SW) begin
                    alu_opcode_temp = ALU_OP_ADD; // SW (Address Calculation)
                end
            end

            // B-type Instructions (e.g., BNE, BLT)
            `OPCODE_BNE: begin
                if (funct3 == `FUNCT3_BNE) begin
                    alu_opcode_temp = ALU_OP_BNE; // BNE
                end
                if (funct3 == `FUNCT3_BLT) begin
                    alu_opcode_temp = ALU_OP_BLT; // BLT
                end
            end
            // U-type Instructions (e.g., LUI, AUIPC)
            `OPCODE_LUI: begin
                alu_opcode_temp = ALU_OP_LUI; // LUI
            end
            `OPCODE_AUIPC: begin
                alu_opcode_temp = ALU_OP_NOP; // AUIPC (Handled in the PC logic)
            end

            // J-type Instructions (e.g., JAL, JALR)
            `OPCODE_JAL: begin
                alu_opcode_temp = ALU_OP_NOP; // JAL (Handled in the control flow logic)
            end
            `OPCODE_JALR: begin
                if (funct3 == `FUNCT3_JALR) begin
                    alu_opcode_temp = ALU_OP_NOP; // JALR (Handled in the control flow logic)
                end
            end

            default: begin
                alu_opcode_temp = ALU_OP_NOP; // Default: NOP
            end
        endcase
    end


    assign operand_1_o     = rs1_dout_i; // Default to rs1 data

    // Declare a temporary reg variable
    reg [REG_DW-1:0] operand_2_temp;

    // Connect the temp variable to the output wire
    assign operand_2_o = operand_2_temp;

    always @(*) begin
        case (opcode)
            // I-type instructions (e.g., ADDI, SLTI, LW)
            7'b000_0011, 7'b001_0011: begin
                operand_2_temp = {{20{imm_i[11]}}, imm_i}; // Sign-extend I-type immediate
            end

            // S-type instructions (e.g., SW)
            `OPCODE_SW: begin
                operand_2_temp = {{20{imm_s[11]}}, imm_s}; // Sign-extend S-type immediate
            end

            // B-type instructions (e.g., BNE, BLT)
            7'b110_0011: begin
                operand_2_temp = rs2_dout_i; // Use rs2 for comparison in branches
            end

            // U-type instructions (e.g., LUI, AUIPC)
            `OPCODE_LUI, `OPCODE_AUIPC: begin
                operand_2_temp = {{imm_u[31:12]}, {12{1'b0}}}; // Upper immediate
            end

            // Default (e.g., R-type instructions)
            default: begin
                operand_2_temp = rs2_dout_i; // Default to rs2 value for R-type
            end
        endcase
    end


    // 鬼才设计，在 decode 里加上比较器
    assign branch_en_o     = (opcode == `OPCODE_BNE && rs1_dout_i != rs2_dout_i) ||
                             (opcode == `OPCODE_BLT && $signed(rs1_dout_i) < $signed(rs2_dout_i));
    assign branch_offset_o = {{19{imm_b[12]}}, imm_b}; // Sign-extended branch offset

    assign jump_en_o       = (opcode == `OPCODE_JAL || opcode == `OPCODE_JALR);

    // decode 也有加法器
    assign jump_offset_o   = (opcode == `OPCODE_JAL) ? {{11{imm_j[20]}}, imm_j} : 
                             (opcode == `OPCODE_JALR) ? (rs1_dout_i + {{20{imm_i[11]}}, imm_i}) :
                             0;

    assign mem_ren_o       = (opcode == `OPCODE_LW);
    assign mem_wen_o       = (opcode == `OPCODE_SW);
    assign mem_din_o       = rs2_dout_i; // SW stores rs2

    assign id_wb_en_o      = (opcode == `OPCODE_ADD || opcode == `OPCODE_ADDI ||
                               opcode == `OPCODE_LW || opcode == `OPCODE_JAL ||
                               opcode == `OPCODE_JALR || opcode == `OPCODE_LUI ||
                               opcode == `OPCODE_AUIPC || opcode == `OPCODE_MUL ||
                               opcode == `OPCODE_SLL || opcode == `OPCODE_AND ||
                               opcode == `OPCODE_SLTI);
    assign id_wb_sel_o     = (opcode == `OPCODE_LW); // 0 for regular, 1 for memory
    assign id_wb_addr_o    = rd; // Write-back destination

endmodule 
