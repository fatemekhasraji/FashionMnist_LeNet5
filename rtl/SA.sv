`timescale 1ns / 1ps

module Torus #(
    parameter bit_width = 32,
    parameter FRAC_BITS = 8,
    parameter mat_size = 5,
    parameter torus_row_size = mat_size,
    parameter torus_col_size = mat_size
)(
    input  wire reset,
    input  wire clk,
    input  wire START,
    input  wire MOVE,
    input  wire MULT_ADD,
    input  wire FINISH,
    input  wire op_sel, // 0=GEMM, 1=POOL
    input  wire signed [(bit_width * mat_size * mat_size) - 1 : 0] flat_A,
    input  wire signed [(bit_width * mat_size * mat_size) - 1 : 0] flat_B,
    output reg  signed [(32 * mat_size * mat_size) - 1 : 0] Matrix_C
);

    reg signed [bit_width-1:0] Matrix_A [0:mat_size - 1][0:mat_size - 1];
    reg signed [bit_width-1:0] Matrix_B [0:mat_size - 1][0:mat_size - 1];

    reg signed [bit_width-1:0] up   [0:torus_row_size-1][0:torus_col_size-1];
    reg signed [bit_width-1:0] left [0:torus_row_size-1][0:torus_col_size-1];
    wire signed [bit_width-1:0] right[0:torus_row_size-1][0:torus_col_size-1];
    wire signed [bit_width-1:0] down [0:torus_row_size-1][0:torus_col_size-1];
    wire signed [31:0] prod [0:torus_row_size-1][0:torus_col_size-1];

    reg signed [bit_width-1:0] up_reg   [0:torus_row_size-1][0:torus_col_size-1];
    reg signed [bit_width-1:0] left_reg [0:torus_row_size-1][0:torus_col_size-1];

    genvar i, j;
    generate
        for (i = 0; i < torus_row_size; i = i + 1) begin : row
            for (j = 0; j < torus_col_size; j = j + 1) begin : col
                PE #(
                    .bit_width(bit_width),
                    .FRAC_BITS(FRAC_BITS),
                    .mat_size(mat_size),
                    .pos_row(i),
                    .pos_col(j)
                ) PE_inst (
                    .reset(reset),
                    .clk(clk),
                    .up(up_reg[i][j]),
                    .left(left_reg[i][j]),
                    .START(START),
                    .MULT_ADD(MULT_ADD),
                    .down(down[i][j]),
                    .right(right[i][j]),
                    .prod(prod[i][j])
                );
            end
        end
    endgenerate

    integer r, c;
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            for (r = 0; r < torus_row_size; r = r + 1)
                for (c = 0; c < torus_col_size; c = c + 1) begin
                    up_reg[r][c]   <= 0;
                    left_reg[r][c] <= 0;
                end
        end else begin
            for (r = 0; r < torus_row_size; r = r + 1)
                for (c = 0; c < torus_col_size; c = c + 1) begin
                    up_reg[r][c]   <= (START) ? up[r][c] :
                                      (MOVE && r != 0) ? down[r-1][c] :
                                      (MOVE && r == 0) ? down[mat_size-1][c] : up_reg[r][c];
                    left_reg[r][c] <= (START) ? left[r][c] :
                                      (MOVE && c != 0) ? right[r][c-1] :
                                      (MOVE && c == 0) ? right[r][mat_size-1] : left_reg[r][c];
                end
        end
    end

    always @(*) begin
        for (r = 0; r < mat_size; r = r + 1)
            for (c = 0; c < mat_size; c = c + 1) begin
                Matrix_A[r][c] = flat_A[(r * mat_size + c) * bit_width +: bit_width];
                Matrix_B[r][c] = flat_B[(r * mat_size + c) * bit_width +: bit_width];
            end
    end

    integer temp;
    always @(*) begin
        if (!reset) begin
            for (r = 0; r < torus_row_size; r = r + 1)
                for (c = 0; c < torus_col_size; c = c + 1) begin
                    left[r][c] = 0;
                    up[r][c]   = 0;
                end
            if (op_sel == 1'b0) begin
                for (r = 0; r < mat_size; r = r + 1)
                    for (c = 0; c < mat_size; c = c + 1) begin
                        temp = (mat_size - ((r + 1 + c) % mat_size)) % mat_size;
                        left[r][c] = Matrix_B[temp][r];
                        up[r][c]   = Matrix_A[c][temp];
                    end
            end
        end
    end

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            Matrix_C <= 0;
        end else if (FINISH) begin
            for (r = 0; r < mat_size; r = r + 1)
                for (c = 0; c < mat_size; c = c + 1)
                    Matrix_C[(c * mat_size + r) * 32 +: 32] <= prod[r][c];
        end
    end
endmodule
