`timescale 1ns / 1ps
// Minimal test: direct Torus instantiation (not through Top_Controller)
// to isolate Controller FSM behavior
module tb_direct_torus;
    parameter BW = 16;
    parameter MS = 5;

    reg clk = 0;
    reg reset = 1;
    reg START = 0;
    reg signed [(BW*MS*MS)-1:0] flat_A, flat_B;
    wire signed [(BW*MS*MS)-1:0] Matrix_C;
    wire FINISH, MOVE, MULT_ADD;

    // Direct Torus + Controller
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
        $display("=== Direct Torus Test ===");

        // Load data (all ones)
        for (r = 0; r < MS; r = r + 1)
            for (c = 0; c < MS; c = c + 1) begin
                flat_A[(r*MS + c)*BW +: BW] = 16'sd256;
                flat_B[(r*MS + c)*BW +: BW] = 16'sd256;
            end

        // Reset: first posedge sees reset=1
        #3;
        $display("T=%0t: pre-reset", $time);
        @(posedge clk);
        $display("T=%0t: reset=%b START=%b", $time, reset, START);
        #2 reset = 0;

        // Pulse START
        @(posedge clk);
        $display("T=%0t: reset=%b START=%b — pulsing START", $time, reset, START);
        START = 1;
        @(posedge clk);
        START = 0;
        $display("T=%0t: START=0", $time);

        // Monitor
        for (i = 0; i < 12; i = i + 1) begin
            @(posedge clk);
            $display("  T=%0t cycle %0d: START=%b MOVE=%b MULT_ADD=%b FINISH=%b",
                     $time, i, START, MOVE, MULT_ADD, FINISH);
        end

        $display("=== DONE ===");
        $finish;
    end
endmodule
