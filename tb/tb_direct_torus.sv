`timescale 1ns / 1ps

module tb_direct_torus;
    parameter BW = 16;
    parameter MS = 5;

    reg clk = 0;
    reg reset = 1;
    reg START = 0;
    reg signed [(BW*MS*MS)-1:0] flat_A, flat_B;
    wire signed [(32*MS*MS)-1:0] Matrix_C;
    wire FINISH, MOVE, MULT_ADD;

    wire ctrl_MOVE, ctrl_MULT_ADD, ctrl_FINISH;

    Controller #(.MAT_SIZE(MS)) ctrl (
        .clk(clk), .reset(reset), .START(START),
        .op_sel(1'b0),
        .in_rows(MS[2:0]), .in_cols(MS[2:0]),
        .out_rows(MS[2:0]), .out_cols(MS[2:0]),
        .k_dim(MS[2:0]),
        .MOVE(ctrl_MOVE), .MULT_ADD(ctrl_MULT_ADD),
        .FINISH(ctrl_FINISH), .POOL_MODE()
    );

    Torus #(.bit_width(BW), .mat_size(MS), .torus_row_size(MS), .torus_col_size(MS))
    sa (
        .reset(reset), .clk(clk), .START(START),
        .MOVE(ctrl_MOVE), .MULT_ADD(ctrl_MULT_ADD),
        .FINISH(ctrl_FINISH),
        .op_sel(1'b0),
        .flat_A(flat_A), .flat_B(flat_B), .Matrix_C(Matrix_C)
    );

    assign FINISH = ctrl_FINISH;
    assign MOVE = ctrl_MOVE;
    assign MULT_ADD = ctrl_MULT_ADD;

    always #5 clk = ~clk;

    integer r, c, i;

    initial begin
        $display("==================================================");
        $display("  Direct Torus Verification");
        $display("==================================================");

        for (r = 0; r < MS; r = r + 1)
            for (c = 0; c < MS; c = c + 1) begin
                flat_A[(r*MS + c)*BW +: BW] = 16'sd256;
                flat_B[(r*MS + c)*BW +: BW] = 16'sd256;
            end

        #3;
        @(posedge clk);
        #2 reset = 0;

        @(posedge clk);
        START = 1;
        @(posedge clk);
        START = 0;

        wait (FINISH);
        @(posedge clk);

        $display("  Torus execution complete. Matrix_C ready.");
        $display("==================================================");
        #20;
        $finish;
    end
endmodule
