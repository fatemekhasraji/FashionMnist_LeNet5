import numpy as np
import os

def read_hex_weights(filename):
    vals = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                v = int(line, 16)
                if v >= 32768: v -= 65536
                vals.append(v)
    return np.array(vals, dtype=np.float64)

def read_hex_images(filename):
    vals = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                v = int(line, 16)
                if v >= 32768: v -= 65536
                vals.append(v)
    return np.array(vals, dtype=np.float64)

def im2col_fast(img, kh=5, kw=5):
    C, H, W = img.shape
    out_h, out_w = H - kh + 1, W - kw + 1
    col = np.zeros((C, kh, kw, out_h, out_w), dtype=img.dtype)
    for y in range(kh):
        y_max = y + out_h
        for x in range(kw):
            x_max = x + out_w
            col[:, y, x, :, :] = img[:, y:y_max, x:x_max]
    col = col.transpose(3, 4, 0, 1, 2).reshape(out_h * out_w, C * kh * kw)
    return col

def run_fp32_sim(img, weights_dict):
    scale = 256.0
    img_q = img[None, :, :] / scale
    wc1 = weights_dict['w_c1'].reshape(6, 25) / scale
    bc1 = weights_dict['b_c1'] / scale
    wc2 = weights_dict['w_c2'].reshape(16, 150) / scale
    bc2 = weights_dict['b_c2'] / scale
    wc3 = weights_dict['w_c3'].reshape(120, 400) / scale
    bc3 = weights_dict['b_c3'] / scale
    wf1 = weights_dict['w_f1'].reshape(84, 120) / scale
    bf1 = weights_dict['b_f1'] / scale
    wf2 = weights_dict['w_f2'].reshape(10, 84) / scale
    bf2 = weights_dict['b_f2'] / scale

    # Conv1
    col1 = im2col_fast(img_q, 5, 5)
    c1 = np.maximum(0, np.dot(col1, wc1.T) + bc1).T.reshape(6, 28, 28)

    # Pool1
    p1 = c1.reshape(6, 14, 2, 14, 2).mean(axis=(2, 4))

    # Conv2
    col2 = im2col_fast(p1, 5, 5)
    c2 = np.maximum(0, np.dot(col2, wc2.T) + bc2).T.reshape(16, 10, 10)

    # Pool2
    p2 = c2.reshape(16, 5, 2, 5, 2).mean(axis=(2, 4))

    # Conv3
    col3 = p2.reshape(1, 400)
    c3 = np.maximum(0, np.dot(col3, wc3.T) + bc3).reshape(120)

    # FC1
    f1 = np.maximum(0, np.dot(wf1, c3) + bf1)

    # FC2
    f2 = np.dot(wf2, f1) + bf2

    return np.argmax(f2)

def run_quantized_sim(img, weights_dict, frac_bits=8, word_bits=16):
    scale = 1 << frac_bits
    max_val = (1 << (word_bits - 1)) - 1
    min_val = -(1 << (word_bits - 1))
    clamp = lambda x: np.clip(x, min_val, max_val)
    asr = lambda x, s: np.right_shift(x.astype(np.int64), s)

    img_q = clamp(np.round((img[None, :, :] / 256.0) * scale)).astype(np.int64)
    wc1 = clamp(np.round((weights_dict['w_c1'].reshape(6, 25) / 256.0) * scale)).astype(np.int64)
    bc1 = clamp(np.round((weights_dict['b_c1'] / 256.0) * scale)).astype(np.int64)
    wc2 = clamp(np.round((weights_dict['w_c2'].reshape(16, 150) / 256.0) * scale)).astype(np.int64)
    bc2 = clamp(np.round((weights_dict['b_c2'] / 256.0) * scale)).astype(np.int64)
    wc3 = clamp(np.round((weights_dict['w_c3'].reshape(120, 400) / 256.0) * scale)).astype(np.int64)
    bc3 = clamp(np.round((weights_dict['b_c3'] / 256.0) * scale)).astype(np.int64)
    wf1 = clamp(np.round((weights_dict['w_f1'].reshape(84, 120) / 256.0) * scale)).astype(np.int64)
    bf1 = clamp(np.round((weights_dict['b_f1'] / 256.0) * scale)).astype(np.int64)
    wf2 = clamp(np.round((weights_dict['w_f2'].reshape(10, 84) / 256.0) * scale)).astype(np.int64)
    bf2 = clamp(np.round((weights_dict['b_f2'] / 256.0) * scale)).astype(np.int64)

    # Conv1
    col1 = im2col_fast(img_q, 5, 5)
    c1_mat = asr(np.dot(col1, wc1.T), frac_bits) + bc1
    c1 = clamp(np.maximum(0, c1_mat)).T.reshape(6, 28, 28)

    # Pool1 (2x2 avg)
    p1 = clamp(c1.reshape(6, 14, 2, 14, 2).mean(axis=(2, 4)).astype(np.int64))

    # Conv2
    col2 = im2col_fast(p1, 5, 5)
    c2_mat = asr(np.dot(col2, wc2.T), frac_bits) + bc2
    c2 = clamp(np.maximum(0, c2_mat)).T.reshape(16, 10, 10)

    # Pool2 (2x2 avg)
    p2 = clamp(c2.reshape(16, 5, 2, 5, 2).mean(axis=(2, 4)).astype(np.int64))

    # Conv3
    col3 = p2.reshape(1, 400)
    c3_mat = asr(np.dot(col3, wc3.T), frac_bits) + bc3
    c3 = clamp(np.maximum(0, c3_mat)).reshape(120)

    # FC1
    f1_mat = asr(np.dot(wf1, c3), frac_bits) + bf1
    f1 = clamp(np.maximum(0, f1_mat))

    # FC2
    f2_mat = asr(np.dot(wf2, f1), frac_bits) + bf2
    f2 = clamp(f2_mat)

    return np.argmax(f2)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    wdir = os.path.join(base_dir, 'weights')
    weights_dict = {
        'w_c1': read_hex_weights(os.path.join(wdir, 'conv1_weight.txt')),
        'b_c1': read_hex_weights(os.path.join(wdir, 'conv1_bias.txt')),
        'w_c2': read_hex_weights(os.path.join(wdir, 'conv2_weight.txt')),
        'b_c2': read_hex_weights(os.path.join(wdir, 'conv2_bias.txt')),
        'w_c3': read_hex_weights(os.path.join(wdir, 'conv3_weight.txt')),
        'b_c3': read_hex_weights(os.path.join(wdir, 'conv3_bias.txt')),
        'w_f1': read_hex_weights(os.path.join(wdir, 'fc1_weight.txt')),
        'b_f1': read_hex_weights(os.path.join(wdir, 'fc1_bias.txt')),
        'w_f2': read_hex_weights(os.path.join(wdir, 'fc2_weight.txt')),
        'b_f2': read_hex_weights(os.path.join(wdir, 'fc2_bias.txt'))
    }

    images = read_hex_images(os.path.join(base_dir, 'verilog', 'test_data', 'test_images.hex')).reshape(20, 32, 32)
    labels = [9, 2, 1, 1, 6, 1, 4, 6, 5, 7, 4, 5, 7, 3, 4, 1, 2, 4, 8, 0]

    correct_fp32 = 0
    correct_q88 = 0
    correct_q44 = 0
    n_samples = len(labels)

    for i in range(n_samples):
        img = images[i]
        lbl = labels[i]

        pred_fp32 = run_fp32_sim(img, weights_dict)
        pred_q88  = run_quantized_sim(img, weights_dict, frac_bits=8, word_bits=16)
        pred_q44  = run_quantized_sim(img, weights_dict, frac_bits=4, word_bits=8)

        if pred_fp32 == lbl: correct_fp32 += 1
        if pred_q88 == lbl: correct_q88 += 1
        if pred_q44 == lbl: correct_q44 += 1

    print("="*65)
    print(" QUANTIZATION SENSITIVITY & ACCURACY STUDY (FASHION-MNIST)")
    print("="*65)
    print(f"1. FP32 Floating-Point Baseline : {correct_fp32}/{n_samples} ({correct_fp32/n_samples*100:.1f}%)")
    print(f"2. Q8.8 Fixed-Point (16-bit) RTL : {correct_q88}/{n_samples} ({correct_q88/n_samples*100:.1f}%)")
    print(f"3. Q4.4 Fixed-Point (8-bit) INT8 : {correct_q44}/{n_samples} ({correct_q44/n_samples*100:.1f}%)")
    print("="*65)

if __name__ == '__main__':
    main()
