`timescale 1ns / 1ps

module accel_top #(
    parameter integer DATA_W = 16,
    parameter integer FRAC_BITS = 8,
    parameter integer SA_SIZE = 5
)(
    input  wire clk,
    input  wire reset,
    input  wire START,
    input  wire op_sel, // 0=GEMM, 1=POOL
    input  wire signed [(DATA_W * SA_SIZE * SA_SIZE) - 1 : 0] flat_A,
    input  wire signed [(DATA_W * SA_SIZE * SA_SIZE) - 1 : 0] flat_B,
    output wire signed [(32 * SA_SIZE * SA_SIZE) - 1 : 0] Matrix_C,
    output wire FINISH,
    output wire busy
);
    wire ctrl_MOVE;
    wire ctrl_MULT_ADD;
    wire ctrl_FINISH;

    assign FINISH = ctrl_FINISH;
    assign busy   = !ctrl_FINISH;

    Controller #(
        .MAT_SIZE(SA_SIZE)
    ) ctrl_inst (
        .clk(clk),
        .reset(reset),
        .START(START),
        .op_sel(op_sel),
        .in_rows(SA_SIZE[2:0]),
        .in_cols(SA_SIZE[2:0]),
        .out_rows(SA_SIZE[2:0]),
        .out_cols(SA_SIZE[2:0]),
        .k_dim(SA_SIZE[2:0]),
        .MOVE(ctrl_MOVE),
        .MULT_ADD(ctrl_MULT_ADD),
        .FINISH(ctrl_FINISH),
        .POOL_MODE()
    );

    Torus #(
        .bit_width(DATA_W),
        .FRAC_BITS(FRAC_BITS),
        .mat_size(SA_SIZE),
        .torus_row_size(SA_SIZE),
        .torus_col_size(SA_SIZE)
    ) sa_inst (
        .reset(reset),
        .clk(clk),
        .START(START),
        .MOVE(ctrl_MOVE),
        .MULT_ADD(ctrl_MULT_ADD),
        .FINISH(ctrl_FINISH),
        .op_sel(op_sel),
        .flat_A(flat_A),
        .flat_B(flat_B),
        .Matrix_C(Matrix_C)
    );

endmodule
