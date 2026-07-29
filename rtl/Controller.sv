`timescale 1ns / 1ps

module Controller #(
    parameter MAT_SIZE = 5
)(
    input  wire clk,
    input  wire reset,
    input  wire START,
    input  wire op_sel, // 0=GEMM, 1=POOL
    input  wire [2:0] in_rows,
    input  wire [2:0] in_cols,
    input  wire [2:0] out_rows,
    input  wire [2:0] out_cols,
    input  wire [2:0] k_dim,
    output wire MOVE,
    output wire MULT_ADD,
    output wire FINISH,
    output wire POOL_MODE
);

    typedef enum logic [2:0] {
        S_IDLE  = 3'd0,
        S_PRIME = 3'd1,
        S_RUN   = 3'd2,
        S_DONE  = 3'd3
    } state_t;

    state_t state;
    reg [$clog2(MAT_SIZE+1)-1:0] step_cnt;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= S_IDLE;
            step_cnt <= '0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (START) begin
                        state <= S_PRIME;
                        step_cnt <= '0;
                    end
                end

                S_PRIME: begin
                    if (op_sel) begin
                        state <= S_DONE;
                    end else begin
                        step_cnt <= 1;
                        if (MAT_SIZE <= 1)
                            state <= S_DONE;
                        else
                            state <= S_RUN;
                    end
                end

                S_RUN: begin
                    if (op_sel) begin
                        state <= S_DONE;
                    end else begin
                        step_cnt <= step_cnt + 1;
                        if (step_cnt >= MAT_SIZE[$clog2(MAT_SIZE+1)-1:0])
                            state <= S_DONE;
                    end
                end

                S_DONE: begin
                    if (!START)
                        state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // Combinational outputs
    assign MULT_ADD  = (!op_sel && (state == S_PRIME || state == S_RUN));
    assign MOVE      = (!op_sel && state == S_RUN);
    assign FINISH    = (state == S_DONE);
    assign POOL_MODE = op_sel;

endmodule
