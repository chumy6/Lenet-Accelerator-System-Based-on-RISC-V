`timescale 1ns / 1ps

module im2col #(
    parameter IMG_C         = 1,
    parameter IMG_W         = 8,
    parameter IMG_H         = 8,
    parameter DATA_WIDTH    = 8,
    parameter ADDR_WIDTH    = 32,
    parameter FILTER_SIZE   = 3,
    parameter IMG_BASE      = 16'h0000,
    parameter IM2COL_BASE   = 16'h2000
) (
    input clk,
    input rst_n,
    input [DATA_WIDTH-1:0] data_rd,
    output [DATA_WIDTH-1:0] data_wr,
    output [ADDR_WIDTH-1:0] addr_wr,
    output [ADDR_WIDTH-1:0] addr_rd,
    output reg done,
    output reg mem_wr_en
);

    // Internal registers and signals
    reg [ADDR_WIDTH-1:0] addr_rd_reg;
    reg [ADDR_WIDTH-1:0] addr_wr_reg;
    reg [DATA_WIDTH-1:0] data_wr_reg;
    reg [3:0] state;
    reg [11:0] row_counter;
    reg [11:0] col_counter;
    reg [11:0] filter_row_counter;
    reg [11:0] filter_col_counter;
    reg [11:0] channel_counter;
    reg [11:0] write_counter;

    localparam IDLE = 4'b0000;
    localparam READ = 4'b0001;
    localparam WRITE = 4'b0010;
    localparam DONE = 4'b0011;

    parameter idx1 = $clog2((2+IMG_H) * (2+IMG_W) * IMG_C) - 1;
    parameter idx2 = $clog2((FILTER_SIZE * FILTER_SIZE * IMG_C) * ((2+IMG_H) * (2+IMG_W))) - 1;
    
    wire is3x3 = FILTER_SIZE == 3;
    wire[11:0] padding = is3x3 ? 12'b000000000001 : 12'b000000000000;
    // reg [DATA_WIDTH] mat [(1+IMG_H) * (1+IMG_W) * IMG_C];
    reg [DATA_WIDTH-1:0] mat [(2+IMG_H) * (2+IMG_W) * IMG_C];
    reg [DATA_WIDTH-1:0] transform [(FILTER_SIZE * FILTER_SIZE * IMG_C) * ((2+IMG_H) * (2+IMG_W))];
    reg [11:0] mat_idx;
    wire [11:0] idx;
    assign idx = row_counter * (FILTER_SIZE * FILTER_SIZE * IMG_C) + col_counter;
    assign mem_wr_en = (state == WRITE);

    assign addr_rd = addr_rd_reg;
    assign addr_wr = addr_wr_reg;
    assign data_wr = data_wr_reg;


    // read one data at a time
    // so the im2col module need to ask top module for data (using data_rd)
    // the communication process goes like this.
    //    img2col module send a readresq(addr_wr) to top.v
    //    top.v module gets the data and sends img2col data.
    //    img2col module gets the data, calculate the transformed index, and sends top.v its writeresq(data_wr, addr_rd)
    //    
    // State transition logic

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            row_counter <= 0;
            col_counter <= 0;
            filter_row_counter <= 0;
            filter_col_counter <= 0;
            channel_counter <= 0;
            write_counter <= 0;
            done <= 0;
        end else begin
            case (state)
               IDLE: state <= READ;
               READ:
                    begin
                        // need to add padding to adjust the location
                        mat_idx <= (IMG_W + 2 * padding) * (row_counter+padding) + (col_counter+padding) + channel_counter * 
                                ((IMG_H + 2 * padding) * (IMG_W + 2 * padding));
                        // $display("mat_idx = %d", mat_idx);
                        mat[mat_idx[idx1: 0]] <= (mat_idx[idx1: 0] == 0) && (padding == 1) ? 0 : data_rd;
                        if (channel_counter + 1 == IMG_C) begin
                            channel_counter <= 0;
                            if (col_counter + 1 == IMG_W) begin
                                col_counter <= 0;
                                if (row_counter + 1 == IMG_H) begin
                                    row_counter <= 0;
                                end else begin
                                    row_counter <= row_counter + 1;
                                end
                            end else begin
                                col_counter <= col_counter + 1;
                            end
                        end else begin
                            channel_counter <= channel_counter + 1;
                        end
                        if (addr_rd_reg - 1 == IMG_C * IMG_H * IMG_W) begin
                            row_counter <= 0;
                            channel_counter <= 0;
                            col_counter <= 0;
                            addr_wr_reg <= IM2COL_BASE - 1;
                            mat_idx <= 0;
                            // Transform the data into column-major format
                            for (int h = 0; h < IMG_H; ++h) begin
                                for (int w = 0; w < IMG_W; ++w) begin
                                // For each unit in the output matrix, it results from a convolution, which we need to converts into a dot product of two vector
                                    int row = h * IMG_W + w;
                                    // one row of transformed matrix represents C reception fields flatterned and concatenate together 
                                    for (int c = 0; c < IMG_C; ++c) begin
                                        for (int fh = 0; fh < FILTER_SIZE; ++fh) begin
                                            for (int fw = 0; fw < FILTER_SIZE; ++fw) begin
                                                int ih = h + fh;
                                                int iw = w + fw;
                                                transform
                                                [
                                                 row * (FILTER_SIZE * FILTER_SIZE * IMG_C)
                                                 + c * FILTER_SIZE * FILTER_SIZE + fh * FILTER_SIZE + fw
                                                ]
                                                = mat[c * ((IMG_H + 2 * padding) * (IMG_W + 2 * padding)) + ih * (IMG_W + 2 *padding) + iw];
                                            end
                                        end
                                    end
                                end
                            end
                            state <= WRITE;
                        end else begin
                            addr_rd_reg <=  IMG_BASE + addr_rd_reg + 1;
                        end
                    end
               WRITE:
                begin
                    if (row_counter + 1 == IMG_H * IMG_W) begin
                        row_counter <= 0;
                        if (col_counter + 1 == FILTER_SIZE * FILTER_SIZE * IMG_C) begin
                        end else begin
                            col_counter <= col_counter + 1;
                        end
                    end else begin
                        row_counter <= row_counter + 1;
                    end
                    data_wr_reg <= transform[idx[idx2:0]];
                    if (addr_wr_reg == FILTER_SIZE * FILTER_SIZE * IMG_C *
                        IMG_H * IMG_W + IM2COL_BASE)
                    begin
                        state <= DONE;
                    end else begin
                        addr_wr_reg <= addr_wr_reg + 1;
                    end
                end
               DONE: done <= 1;
               default: state <= IDLE;
            endcase
        end
    end

endmodule
