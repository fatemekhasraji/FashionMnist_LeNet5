`timescale 1ns / 1ps

// ============================================================
// LeNet-5 CNN Hardware Accelerator Top Module
// Datapath sequencing: Conv1 -> Pool1 -> Conv2 -> Pool2 -> Conv3 -> FC1 -> FC2 -> Argmax
// Accelerates 5x5 GEMM tiles using Torus Systolic Array & Controller
// ============================================================

module find_max_index #(
    parameter integer N = 10,
    parameter integer W = 16
)(
    input  wire signed [N*W-1:0] vec,
    output reg  [$clog2(N)-1:0]   idx,
    output reg  signed [W-1:0]    max_val
);
    integer i;
    reg signed [W-1:0] cur;

    always @(*) begin
        max_val = vec[0 +: W];
        idx     = 0;
        for (i = 1; i < N; i = i + 1) begin
            cur = vec[i*W +: W];
            if (cur > max_val) begin
                max_val = cur;
                idx = i[$clog2(N)-1:0];
            end
        end
    end
endmodule

module cnn_top #(
    parameter integer DATA_W = 16,
    parameter integer FRAC_BITS = 8,
    parameter integer SA_SIZE = 5
)(
    input  wire clk,
    input  wire reset,
    input  wire START,
    input  wire signed [DATA_W*32*32-1:0] image_flat,
    output reg  busy,
    output reg  done,
    output reg  [2:0] state_dbg,
    output reg  [2:0] layer_dbg,
    output reg  [3:0] predicted_class,
    output reg signed [DATA_W*10-1:0] logits_flat
);

    // -------------------------
    // Layer dimensions
    // -------------------------
    localparam integer IMG_H   = 32;
    localparam integer IMG_W   = 32;
    localparam integer C1_CH   = 6;
    localparam integer C1_H    = 28;
    localparam integer C1_W    = 28;
    localparam integer P1_H    = 14;
    localparam integer P1_W    = 14;
    localparam integer C2_CH   = 16;
    localparam integer C2_H    = 10;
    localparam integer C2_W    = 10;
    localparam integer P2_H    = 5;
    localparam integer P2_W    = 5;
    localparam integer C3_CH   = 120;
    localparam integer FC1_CH  = 84;
    localparam integer FC2_CH  = 10;

    localparam integer C1_N = C1_H * C1_W; // 784
    localparam integer C2_N = C2_H * C2_W; // 100
    localparam integer C3_N = 1;
    localparam integer FC1_N = 1;
    localparam integer FC2_N = 1;

    // -------------------------
    // Module-local memories
    // -------------------------
    reg signed [DATA_W-1:0] img      [0:IMG_H*IMG_W-1];
    reg signed [DATA_W-1:0] conv1_out[0:C1_CH*C1_H*C1_W-1];
    reg signed [DATA_W-1:0] pool1_out[0:C1_CH*P1_H*P1_W-1];
    reg signed [DATA_W-1:0] conv2_out[0:C2_CH*C2_H*C2_W-1];
    reg signed [DATA_W-1:0] pool2_out[0:C2_CH*P2_H*P2_W-1];
    reg signed [DATA_W-1:0] conv3_out[0:C3_CH-1];
    reg signed [DATA_W-1:0] fc1_out  [0:FC1_CH-1];
    reg signed [DATA_W-1:0] fc2_out  [0:FC2_CH-1];

    // Weights and biases loaded from data/weights/
    reg signed [DATA_W-1:0] w_conv1 [0:C1_CH*25-1];   // 6 x 25
    reg signed [DATA_W-1:0] b_conv1 [0:C1_CH-1];
    reg signed [DATA_W-1:0] w_conv2 [0:C2_CH*150-1];  // 16 x 150
    reg signed [DATA_W-1:0] b_conv2 [0:C2_CH-1];
    reg signed [DATA_W-1:0] w_conv3 [0:C3_CH*400-1];  // 120 x 400
    reg signed [DATA_W-1:0] b_conv3 [0:C3_CH-1];
    reg signed [DATA_W-1:0] w_fc1   [0:FC1_CH*120-1];
    reg signed [DATA_W-1:0] b_fc1   [0:FC1_CH-1];
    reg signed [DATA_W-1:0] w_fc2   [0:FC2_CH*84-1];
    reg signed [DATA_W-1:0] b_fc2   [0:FC2_CH-1];

    // 5x5 torus microkernel interface
    reg  micro_start;
    reg  [1:0] micro_mode;
    reg  micro_op_sel; // 0=GEMM
    reg  [2:0] micro_in_rows, micro_in_cols, micro_out_rows, micro_out_cols, micro_k_dim;
    reg  signed [DATA_W*SA_SIZE*SA_SIZE-1:0] micro_flat_A;
    reg  signed [DATA_W*SA_SIZE*SA_SIZE-1:0] micro_flat_B;
    wire signed [32*SA_SIZE*SA_SIZE-1:0] micro_flat_C;
    wire micro_MOVE, micro_MULT_ADD, micro_FINISH, micro_POOL_MODE;
    integer img_count = 0;

    Controller micro_ctrl (
        .clk(clk),
        .reset(reset),
        .START(micro_start),
        .op_sel(micro_op_sel),
        .in_rows(micro_in_rows),
        .in_cols(micro_in_cols),
        .out_rows(micro_out_rows),
        .out_cols(micro_out_cols),
        .k_dim(micro_k_dim),
        .MOVE(micro_MOVE),
        .MULT_ADD(micro_MULT_ADD),
        .FINISH(micro_FINISH),
        .POOL_MODE(micro_POOL_MODE)
    );

    Torus #(
        .bit_width(DATA_W),
        .FRAC_BITS(FRAC_BITS),
        .mat_size(SA_SIZE),
        .torus_row_size(SA_SIZE),
        .torus_col_size(SA_SIZE)
    ) micro_torus (
        .reset(reset),
        .clk(clk),
        .START(micro_start),
        .MOVE(micro_MOVE),
        .MULT_ADD(micro_MULT_ADD),
        .FINISH(micro_FINISH),
        .op_sel(1'b0),
        .flat_A(micro_flat_A),
        .flat_B(micro_flat_B),
        .Matrix_C(micro_flat_C)
    );

    // 5x5 tile scratchpads
    reg signed [DATA_W-1:0] a_tile [0:SA_SIZE-1][0:SA_SIZE-1];
    reg signed [DATA_W-1:0] b_tile [0:SA_SIZE-1][0:SA_SIZE-1];
    reg signed [31:0]       p_tile [0:SA_SIZE-1][0:SA_SIZE-1];
    reg signed [31:0]       acc_tile [0:SA_SIZE-1][0:SA_SIZE-1];

    integer i, j, rr, cc, kk, n0, m0, k0;
    integer oy, ox, ky, kx, ic, rem;
    integer sum_i;
    integer abs_i;
    integer q_i;

    function automatic signed [DATA_W-1:0] clamp_act_FB(input integer x);
        integer limit;
        begin
            limit = 1 << FRAC_BITS;
            if (x > limit) clamp_act_FB = limit;
            else if (x < -limit) clamp_act_FB = -limit;
            else clamp_act_FB = x;
        end
    endfunction

    function automatic signed [DATA_W-1:0] sat_16(input integer x);
        begin
            if (x > 32767) sat_16 = 16'sd32767;
            else if (x < -32768) sat_16 = -16'sd32768;
            else sat_16 = x;
        end
    endfunction

    function automatic signed [DATA_W-1:0] get_bias(input integer layer, input integer out_ch);
        begin
            case (layer)
                0: get_bias = b_conv1[out_ch];
                2: get_bias = b_conv2[out_ch];
                4: get_bias = b_conv3[out_ch];
                5: get_bias = b_fc1[out_ch];
                6: get_bias = b_fc2[out_ch];
                default: get_bias = '0;
            endcase
        end
    endfunction

    function automatic signed [DATA_W-1:0] get_weight(input integer layer, input integer out_ch, input integer k_idx);
        begin
            case (layer)
                0: get_weight = w_conv1[out_ch*25   + k_idx];
                2: get_weight = w_conv2[out_ch*150  + k_idx];
                4: get_weight = w_conv3[out_ch*400  + k_idx];
                5: get_weight = w_fc1[out_ch*120    + k_idx];
                6: get_weight = w_fc2[out_ch*84     + k_idx];
                default: get_weight = '0;
            endcase
        end
    endfunction

    function automatic signed [DATA_W-1:0] get_a(input integer layer, input integer n_idx, input integer k_idx);
        integer oy, ox, ky, kx, ic, rem;
        begin
            case (layer)
                0: begin
                    oy = n_idx / C1_W;
                    ox = n_idx % C1_W;
                    ky = k_idx / 5;
                    kx = k_idx % 5;
                    get_a = img[(oy + ky) * IMG_W + (ox + kx)];
                end
                2: begin
                    oy  = n_idx / C2_W;
                    ox  = n_idx % C2_W;
                    ic  = k_idx / 25;
                    rem = k_idx % 25;
                    ky  = rem / 5;
                    kx  = rem % 5;
                    get_a = pool1_out[ic * (P1_H * P1_W) + (oy + ky) * P1_W + (ox + kx)];
                end
                4: begin
                    ic  = k_idx / 25;
                    rem = k_idx % 25;
                    ky  = rem / 5;
                    kx  = rem % 5;
                    get_a = pool2_out[ic * (P2_H * P2_W) + ky * P2_W + kx];
                end
                5: get_a = conv3_out[k_idx];
                6: get_a = fc1_out[k_idx];
                default: get_a = '0;
            endcase
        end
    endfunction

    initial begin
        $readmemh("data/weights/conv1_weight.txt", w_conv1);
        $readmemh("data/weights/conv1_bias.txt",   b_conv1);
        $readmemh("data/weights/conv2_weight.txt", w_conv2);
        $readmemh("data/weights/conv2_bias.txt",   b_conv2);
        $readmemh("data/weights/conv3_weight.txt", w_conv3);
        $readmemh("data/weights/conv3_bias.txt",   b_conv3);
        $readmemh("data/weights/fc1_weight.txt",   w_fc1);
        $readmemh("data/weights/fc1_bias.txt",     b_fc1);
        $readmemh("data/weights/fc2_weight.txt",   w_fc2);
        $readmemh("data/weights/fc2_bias.txt",     b_fc2);
    end

    task automatic unpack_image;
        integer idx;
        begin
            for (idx = 0; idx < IMG_H*IMG_W; idx = idx + 1) begin
                img[idx] = image_flat[idx*DATA_W +: DATA_W];
            end
        end
    endtask

    task automatic pack_logits;
        integer idx;
        begin
            logits_flat = '0;
            for (idx = 0; idx < FC2_CH; idx = idx + 1) begin
                logits_flat[idx*DATA_W +: DATA_W] = fc2_out[idx];
            end
        end
    endtask

    task automatic run_microkernel_tile;
        integer r, c;
        begin
            for (r = 0; r < SA_SIZE; r = r + 1) begin
                for (c = 0; c < SA_SIZE; c = c + 1) begin
                    micro_flat_A[(r*SA_SIZE + c)*DATA_W +: DATA_W] = a_tile[r][c];
                    micro_flat_B[(r*SA_SIZE + c)*DATA_W +: DATA_W] = b_tile[r][c];
                end
            end

            micro_start = 1'b1;
            @(posedge clk);
            micro_start <= 1'b0;

            wait (micro_FINISH);
            @(posedge clk);
            @(posedge clk);

            for (r = 0; r < SA_SIZE; r = r + 1) begin
                for (c = 0; c < SA_SIZE; c = c + 1) begin
                    p_tile[r][c] = micro_flat_C[(c*SA_SIZE + r)*32 +: 32];
                end
            end

            wait (!micro_FINISH);
        end
    endtask

    task automatic run_implicit_gemm(input integer layer, input integer N, input integer K, input integer M);
        integer n_base, m_base, k_base;
        integer n_idx, m_idx, k_idx;
        integer rr, cc;
        integer out_idx;
        integer tile_sum;
        begin
            for (n_base = 0; n_base < N; n_base = n_base + SA_SIZE) begin
                for (m_base = 0; m_base < M; m_base = m_base + SA_SIZE) begin
                    for (rr = 0; rr < SA_SIZE; rr = rr + 1) begin
                        for (cc = 0; cc < SA_SIZE; cc = cc + 1) begin
                            acc_tile[rr][cc] = 0;
                        end
                    end

                    for (k_base = 0; k_base < K; k_base = k_base + SA_SIZE) begin
                        for (rr = 0; rr < SA_SIZE; rr = rr + 1) begin
                            for (cc = 0; cc < SA_SIZE; cc = cc + 1) begin
                                n_idx = n_base + rr;
                                k_idx = k_base + cc;
                                a_tile[rr][cc] = (n_idx < N && k_idx < K) ? get_a(layer, n_idx, k_idx) : '0;
                            end
                        end

                        for (rr = 0; rr < SA_SIZE; rr = rr + 1) begin
                            for (cc = 0; cc < SA_SIZE; cc = cc + 1) begin
                                k_idx = k_base + rr;
                                m_idx = m_base + cc;
                                b_tile[rr][cc] = (k_idx < K && m_idx < M) ? get_weight(layer, m_idx, k_idx) : '0;
                            end
                        end

                        run_microkernel_tile();

                        for (rr = 0; rr < SA_SIZE; rr = rr + 1) begin
                            for (cc = 0; cc < SA_SIZE; cc = cc + 1) begin
                                acc_tile[rr][cc] = acc_tile[rr][cc] + $signed(p_tile[rr][cc]);
                            end
                        end
                    end

                    for (rr = 0; rr < SA_SIZE; rr = rr + 1) begin
                        for (cc = 0; cc < SA_SIZE; cc = cc + 1) begin
                            m_idx = m_base + rr;
                            n_idx = n_base + cc;
                            if (n_idx < N && m_idx < M) begin
                                tile_sum = acc_tile[rr][cc] + get_bias(layer, m_idx);
                                if (layer != 6) begin
                                    tile_sum = clamp_act_FB(tile_sum);
                                end else begin
                                    if (FRAC_BITS > 10)
                                        tile_sum = tile_sum >>> (FRAC_BITS - 10);
                                    tile_sum = sat_16(tile_sum);
                                end

                                case (layer)
                                    0: conv1_out[m_idx * C1_N + n_idx] = tile_sum;
                                    2: conv2_out[m_idx * C2_N + n_idx] = tile_sum;
                                    4: conv3_out[m_idx] = tile_sum;
                                    5: fc1_out[m_idx]   = tile_sum;
                                    6: fc2_out[m_idx]   = tile_sum;
                                    default: ;
                                endcase
                            end
                        end
                    end
                end
            end
        end
    endtask

    task automatic run_pool1;
        integer ch, r, c;
        integer base_r, base_c;
        integer sum;
        begin
            for (ch = 0; ch < C1_CH; ch = ch + 1) begin
                for (r = 0; r < P1_H; r = r + 1) begin
                    for (c = 0; c < P1_W; c = c + 1) begin
                        base_r = r * 2;
                        base_c = c * 2;
                        sum = 0;
                        sum = sum + conv1_out[ch * (C1_H * C1_W) + (base_r + 0) * C1_W + (base_c + 0)];
                        sum = sum + conv1_out[ch * (C1_H * C1_W) + (base_r + 0) * C1_W + (base_c + 1)];
                        sum = sum + conv1_out[ch * (C1_H * C1_W) + (base_r + 1) * C1_W + (base_c + 0)];
                        sum = sum + conv1_out[ch * (C1_H * C1_W) + (base_r + 1) * C1_W + (base_c + 1)];
                        pool1_out[ch * (P1_H * P1_W) + r * P1_W + c] = clamp_act_FB(sum >>> 2);
                    end
                end
            end
        end
    endtask

    task automatic run_pool2;
        integer ch, r, c;
        integer base_r, base_c;
        integer sum;
        begin
            for (ch = 0; ch < C2_CH; ch = ch + 1) begin
                for (r = 0; r < P2_H; r = r + 1) begin
                    for (c = 0; c < P2_W; c = c + 1) begin
                        base_r = r * 2;
                        base_c = c * 2;
                        sum = 0;
                        sum = sum + conv2_out[ch * (C2_H * C2_W) + (base_r + 0) * C2_W + (base_c + 0)];
                        sum = sum + conv2_out[ch * (C2_H * C2_W) + (base_r + 0) * C2_W + (base_c + 1)];
                        sum = sum + conv2_out[ch * (C2_H * C2_W) + (base_r + 1) * C2_W + (base_c + 0)];
                        sum = sum + conv2_out[ch * (C2_H * C2_W) + (base_r + 1) * C2_W + (base_c + 1)];
                        pool2_out[ch * (P2_H * P2_W) + r * P2_W + c] = clamp_act_FB(sum >>> 2);
                    end
                end
            end
        end
    endtask

    task automatic run_inference;
        begin
            state_dbg = 3'd1; layer_dbg = 3'd0;
            run_implicit_gemm(0, C1_N, 25, C1_CH);

            state_dbg = 3'd2; layer_dbg = 3'd1;
            run_pool1();

            state_dbg = 3'd3; layer_dbg = 3'd2;
            run_implicit_gemm(2, C2_N, 150, C2_CH);

            state_dbg = 3'd4; layer_dbg = 3'd3;
            run_pool2();

            state_dbg = 3'd5; layer_dbg = 3'd4;
            run_implicit_gemm(4, C3_N, 400, C3_CH);

            state_dbg = 3'd6; layer_dbg = 3'd5;
            run_implicit_gemm(5, FC1_N, 120, FC1_CH);

            state_dbg = 3'd7; layer_dbg = 3'd6;
            run_implicit_gemm(6, FC2_N, 84, FC2_CH);

            pack_logits();
            img_count = img_count + 1;
            state_dbg = 3'd0;
        end
    endtask

    initial begin
        busy = 1'b0;
        done = 1'b0;
        state_dbg = 3'd0;
        layer_dbg = 3'd0;
        predicted_class = 4'd0;
        logits_flat = '0;
        micro_start = 1'b0;
        micro_mode = 2'b00;
        micro_op_sel = 1'b0;
        micro_in_rows = SA_SIZE;
        micro_in_cols = SA_SIZE;
        micro_out_rows = SA_SIZE;
        micro_out_cols = SA_SIZE;
        micro_k_dim = SA_SIZE;
    end

    always @(posedge reset) begin
        busy <= 1'b0;
        done <= 1'b0;
        predicted_class <= 4'd0;
    end

    always @(posedge START) begin
        if (!busy) begin
            busy <= 1'b1;
            done <= 1'b0;
            unpack_image();
            run_inference();
            busy <= 1'b0;
            done <= 1'b1;
        end
    end

    wire [$clog2(FC2_CH)-1:0] max_idx;
    wire signed [DATA_W-1:0] max_val;
    find_max_index #(
        .N(FC2_CH),
        .W(DATA_W)
    ) argmax (
        .vec(logits_flat),
        .idx(max_idx),
        .max_val(max_val)
    );

    always @(*) begin
        predicted_class = max_idx[3:0];
    end

endmodule
