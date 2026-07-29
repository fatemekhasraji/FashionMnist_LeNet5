`timescale 1ns / 1ps

module PE #(
    parameter integer bit_width = 16,
    parameter integer FRAC_BITS = 8,
    parameter integer mat_size  = 4,
    parameter integer pos_row   = 0,
    parameter integer pos_col   = 0
)(
    input  wire reset, clk,
    input  wire signed [bit_width-1:0] up,
    input  wire signed [bit_width-1:0] left,
    input  wire START,
    input  wire MULT_ADD,
    output wire signed [bit_width-1:0] down,
    output wire signed [bit_width-1:0] right,
    output reg  signed [31:0] prod
);

    reg signed [31:0] psum;

    assign down  = up;
    assign right = left;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            psum  <= 0;
            prod  <= 0;
        end else begin
            if (START) begin
                psum <= 0;
                prod <= 0;
            end else if (MULT_ADD && pos_row < mat_size && pos_col < mat_size) begin
                psum <= psum + (($signed({{16{left[15]}}, left}) * $signed({{16{up[15]}}, up})) >>> FRAC_BITS);
                prod <= psum + (($signed({{16{left[15]}}, left}) * $signed({{16{up[15]}}, up})) >>> FRAC_BITS);
            end else begin
                prod <= psum;
            end
        end
    end

endmodule
