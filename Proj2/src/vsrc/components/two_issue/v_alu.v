// =======================================
// You need to finish this module
// =======================================

module v_alu #(
    parameter SEW       = 32,
    parameter VLMAX     = 8,
    parameter VALUOP_DW = 5,
    parameter VREG_DW   = 256
)(
    input  wire [VALUOP_DW-1:0] valu_opcode_i,
    input  wire [VREG_DW-1:0]   operand_v1_i,
    input  wire [VREG_DW-1:0]   operand_v2_i,
    output reg  [VREG_DW-1:0]   valu_result_o
);

localparam VALU_OP_NOP  = 5'd0;
localparam VALU_OP_VADD = 5'd1;
localparam VALU_OP_VMUL = 5'd2;

// Declare arrays to hold element slices
wire [SEW-1:0] v1_lanes [0:VLMAX-1];
wire [SEW-1:0] v2_lanes [0:VLMAX-1];

generate
  genvar i;
  for (i = 0; i < VLMAX; i = i + 1) begin : GEN_LANES
    // Extract each SEW-bit lane from the 256-bit operands
    assign v1_lanes[i] = operand_v1_i[SEW*i +: SEW];
    assign v2_lanes[i] = operand_v2_i[SEW*i +: SEW];
  end
endgenerate

reg [VREG_DW-1:0] add_result;
reg [VREG_DW-1:0] mul_result;

integer j;
always @(*) begin
    // Compute element-wise results for add and mul
    add_result = {VREG_DW{1'b0}};
    mul_result = {VREG_DW{1'b0}};

    for (j = 0; j < VLMAX; j = j + 1) begin
        add_result[SEW*j +: SEW] = v1_lanes[j] + v2_lanes[j];
        mul_result[SEW*j +: SEW] = v1_lanes[j] * v2_lanes[j];
    end
end

always @(*) begin
    // Select final output based on opcode
    case (valu_opcode_i)
        VALU_OP_VADD: valu_result_o = add_result;
        VALU_OP_VMUL: valu_result_o = mul_result;
        default:      valu_result_o = {VREG_DW{1'b0}}; // NOP or unrecognized
    endcase
end

endmodule
