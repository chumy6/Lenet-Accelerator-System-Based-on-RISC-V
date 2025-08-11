// =======================================
// You need to finish this module
// =======================================

module v_regfile #(
    parameter VREG_DW    = 256,
    parameter VREG_AW    = 5
)(
    input                       clk,
    input                       rst,

    input                       vwb_en_i,
    input       [VREG_AW-1:0]   vwb_addr_i,
    input       [VREG_DW-1:0]   vwb_data_i,

    input                       vs1_en_i,
    input       [VREG_AW-1:0]   vs1_addr_i,
    output reg  [VREG_DW-1:0]   vs1_data_o,

    input                       vs2_en_i,
    input       [VREG_AW-1:0]   vs2_addr_i,
    output reg  [VREG_DW-1:0]   vs2_data_o
);

    // Number of registers
    localparam VREG_COUNT = (1 << VREG_AW);

    // 1) Declare the array of vector registers (each 256 bits wide)
    reg [VREG_DW-1:0] vreg_array [0:VREG_COUNT-1];

    // 2) Optional reset: clear all registers on reset
    integer i;
    always @(posedge clk) begin
        if (rst) begin
            // If you want all registers set to 0 on reset:
            for (i = 0; i < VREG_COUNT; i = i + 1) begin
                vreg_array[i] <= {VREG_DW{1'b0}};
            end
        end else begin
            // 3) Write logic (synchronous write)
            if (vwb_en_i) begin
                // Example: If you want `v0` (register 0) to be read-only or used for masks,
                // you can prevent writes when `vwb_addr_i == 0`. Otherwise, always write.
                // if (vwb_addr_i != 0) begin
                //     vreg_array[vwb_addr_i] <= vwb_data_i;
                // end
                vreg_array[vwb_addr_i] <= vwb_data_i;
            end
        end
    end

    // 4) Read logic (asynchronous or synchronous)
    // Here we implement asynchronous reads:
    always @(*) begin
        // Default to zero if not enabled
        vs1_data_o = {VREG_DW{1'b0}};
        if (vs1_en_i) begin
            vs1_data_o = vreg_array[vs1_addr_i];
        end
    end

    always @(*) begin
        // Default to zero if not enabled
        vs2_data_o = {VREG_DW{1'b0}};
        if (vs2_en_i) begin
            vs2_data_o = vreg_array[vs2_addr_i];
        end
    end

endmodule
