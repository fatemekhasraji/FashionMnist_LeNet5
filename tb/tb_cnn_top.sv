`timescale 1ns / 1ps

module tb_cnn_top;
    localparam integer DATA_W = 16;
    localparam integer IMG_H  = 32;
    localparam integer IMG_W  = 32;
    localparam integer IMG_N  = IMG_H * IMG_W;
    localparam integer N_IMGS = 20;
    localparam integer FRAC_BITS = 8;
    
    reg clk = 0;
    reg reset = 1;
    reg START = 0;

    reg signed [DATA_W*IMG_N-1:0] image_flat;
    wire busy;
    wire done;
    wire [2:0] state_dbg;
    wire [2:0] layer_dbg;
    wire [3:0] predicted_class;
    wire signed [DATA_W*10-1:0] logits_flat;

    reg signed [DATA_W-1:0] all_imgs [0:IMG_N * N_IMGS - 1];
    reg [7:0] label_mem [0:N_IMGS - 1];
    reg [9:0] img_idx;
    integer correct, total, i;

    cnn_top #(
        .DATA_W(DATA_W),
        .FRAC_BITS(FRAC_BITS),
        .SA_SIZE(5)
    ) dut (
        .clk(clk),
        .reset(reset),
        .START(START),
        .image_flat(image_flat),
        .busy(busy),
        .done(done),
        .state_dbg(state_dbg),
        .layer_dbg(layer_dbg),
        .predicted_class(predicted_class),
        .logits_flat(logits_flat)
    );

    always #5 clk = ~clk;

    task pack_image;
        input integer idx;
        integer base;
        begin
            base = idx * IMG_N;
            image_flat = '0;
            for (i = 0; i < IMG_N; i = i + 1)
                image_flat[i*DATA_W +: DATA_W] = all_imgs[base + i];
        end
    endtask

    initial begin
        $readmemh("data/test_images.hex", all_imgs);
        $readmemh("data/labels.hex", label_mem);
        $display("==================================================");
        $display("  LeNet-5 Fashion-MNIST Hardware Inference Test");
        $display("==================================================");

        correct = 0;
        total   = 0;

        reset = 1;
        repeat (2) @(posedge clk);
        reset = 0;
        @(posedge clk);

        for (img_idx = 0; img_idx < N_IMGS; img_idx = img_idx + 1) begin
            pack_image(img_idx);

            @(posedge clk);
            START = 1;
            @(posedge clk);
            START = 0;

            wait(done);
            @(posedge clk);

            total = total + 1;
            if (predicted_class == label_mem[img_idx])
                correct = correct + 1;

            $display("  [Image %2d] GroundTruth=%0d | Predicted=%0d | Status=%s",
                img_idx, label_mem[img_idx], predicted_class,
                (predicted_class == label_mem[img_idx] ? "OK" : "WRONG"));
        end

        $display("==================================================");
        $display("  Accuracy: %0d / %0d (%0d%%)", correct, total, correct * 100 / total);
        $display("==================================================");
        #20;
        $finish;
    end

endmodule
