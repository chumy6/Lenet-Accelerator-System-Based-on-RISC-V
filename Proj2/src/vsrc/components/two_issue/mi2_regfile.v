// =======================================
// You need to finish this module
// =======================================

module mi2_regfile #(
    parameter REG_DW    = 32,
    parameter REG_AW    = 5
)(
    input                       clk,
    input                       rst,

    // Write-back for issue slot #1
    input                       is1_wb_en_i,
    input       [REG_AW-1:0]    is1_wb_addr_i,
    input       [REG_DW-1:0]    is1_wb_data_i,

    // Read port 1 for issue slot #1
    input                       is1_rs1_en_i,
    input       [REG_AW-1:0]    is1_rs1_addr_i,
    output reg  [REG_DW-1:0]    is1_rs1_data_o,

    // Read port 2 for issue slot #1
    input                       is1_rs2_en_i,
    input       [REG_AW-1:0]    is1_rs2_addr_i,
    output reg  [REG_DW-1:0]    is1_rs2_data_o,

    // Additional read port (scalar) for issue slot #2
    // (e.g., used by vector instructions requiring a scalar operand or address)
    input                       is2_rs1_en_i,
    input       [REG_AW-1:0]    is2_rs1_addr_i,
    output reg  [REG_DW-1:0]    is2_rs1_data_o
);

localparam REG_COUNT = (1 << REG_AW);  // e.g., 32 registers if REG_AW=5

  // -------------------------------------------------------------------
  // 1) Register File Array
  // -------------------------------------------------------------------
  reg [REG_DW-1:0] regfile[0:REG_COUNT-1];

  // -------------------------------------------------------------------
  // 2) Optional Reset Logic
  // -------------------------------------------------------------------
  // If you'd like all registers to be zero on reset, uncomment and modify:
  integer i;
  always @(posedge clk) begin
    if (rst) begin
      // Reset all registers to 0 (optional)
      for (i = 0; i < REG_COUNT; i = i + 1) begin
        regfile[i] <= {REG_DW{1'b0}};
      end
    end else begin
      // -------------------------------------------------------------------
      // 3) Synchronous Write
      // -------------------------------------------------------------------
      if (is1_wb_en_i) begin
        // If you want x0 to remain 0, skip writing if is1_wb_addr_i == 0
        if (is1_wb_addr_i != 0) begin
          regfile[is1_wb_addr_i] <= is1_wb_data_i;
        end
        regfile[is1_wb_addr_i] <= is1_wb_data_i;
      end
    end
  end

  // -------------------------------------------------------------------
  // 4) Asynchronous Read
  // -------------------------------------------------------------------
  always @(*) begin
    // Default to 0
    is1_rs1_data_o = {REG_DW{1'b0}};
    if (is1_rs1_en_i) begin
      is1_rs1_data_o = (is1_rs1_addr_i == 0) ? {REG_DW{1'b0}} : regfile[is1_rs1_addr_i];
    end
  end

  always @(*) begin
    is1_rs2_data_o = {REG_DW{1'b0}};
    if (is1_rs2_en_i) begin
      is1_rs2_data_o = (is1_rs2_addr_i == 0) ? {REG_DW{1'b0}} : regfile[is1_rs2_addr_i];
    end
  end

  always @(*) begin
    is2_rs1_data_o = {REG_DW{1'b0}};
    if (is2_rs1_en_i) begin
      is2_rs1_data_o = (is2_rs1_addr_i == 0) ? {REG_DW{1'b0}} : regfile[is2_rs1_addr_i];
    end
  end

endmodule

