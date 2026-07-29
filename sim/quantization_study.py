"""
Quantization Sensitivity & Accuracy Trade-off Study
===================================================
Benchmarking FP32 Floating-Point, Q8.8 Fixed-Point, and Q4.4 INT8 Precision.
"""
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

    w_c1 = weights_dict['w_c1'].reshape(6, 25) / scale
    b_c1 = weights_dict['b_c1'] / scale
    w_c2 = weights_dict['w_c2'].reshape(16, 150) / scale
    b_c2 = weights_dict['b_c2'] / scale
    w_c3 = weights_dict['w_c3'].reshape(120, 400) / scale
    b_c3 = weights_dict['b_c3'] / scale
    w_f1 = weights_dict['w_f1'].reshape(84, 120) / scale
    b_f1 = weights_dict['b_f1'] / scale
    w_f2 = weights_dict['w_f2'].reshape(10, 84) / scale
    b_f2 = weights_dict['b_f2'] / scale

    # Conv1
    c1_in = im2col_fast(img_q, 5, 5)
    c1 = (c1_in @ w_c1.T + b_c1).reshape(28, 28, 6).transpose(2, 0, 1)
    c1 = np.maximum(0, c1)

    # Pool1
    p1 = c1.reshape(6, 14, 2, 14, 2).mean(axis=(2, 4))

    # Conv2
    c2_in = im2col_fast(p1, 5, 5)
    c2 = (c2_in @ w_c2.T + b_c2).reshape(10, 10, 16).transpose(2, 0, 1)
    c2 = np.maximum(0, c2)

    # Pool2
    p2 = c2.reshape(16, 5, 2, 5, 2).mean(axis=(2, 4))

    # Conv3
    c3_in = p2.flatten()[None, :]
    c3 = (c3_in @ w_c3.T + b_c3).flatten()
    c3 = np.maximum(0, c3)

    # FC1
    fc1 = (c3[None, :] @ w_f1.T + b_f1).flatten()
    fc1 = np.maximum(0, fc1)

    # FC2
    fc2 = (fc1[None, :] @ w_f2.T + b_f2).flatten()
    return np.argmax(fc2)

def run_quantized_sim(img, weights_dict, frac_bits=8, word_bits=16):
    shift = frac_bits
    max_val = (1 << (word_bits - 1)) - 1
    min_val = -(1 << (word_bits - 1))

    img_q = np.clip(img[None, :, :], min_val, max_val).astype(np.int64)

    w_c1 = weights_dict['w_c1'].reshape(6, 25).astype(np.int64)
    b_c1 = weights_dict['b_c1'].astype(np.int64)
    w_c2 = weights_dict['w_c2'].reshape(16, 150).astype(np.int64)
    b_c2 = weights_dict['b_c2'].astype(np.int64)
    w_c3 = weights_dict['w_c3'].reshape(120, 400).astype(np.int64)
    b_c3 = weights_dict['b_c3'].astype(np.int64)
    w_f1 = weights_dict['w_f1'].reshape(84, 120).astype(np.int64)
    b_f1 = weights_dict['b_f1'].astype(np.int64)
    w_f2 = weights_dict['w_f2'].reshape(10, 84).astype(np.int64)
    b_f2 = weights_dict['b_f2'].astype(np.int64)

    # Conv1
    c1_in = im2col_fast(img_q, 5, 5).astype(np.int64)
    c1_raw = (c1_in @ w_c1.T) >> shift
    c1 = np.clip(c1_raw + b_c1, 0, max_val).reshape(28, 28, 6).transpose(2, 0, 1)

    # Pool1
    p1 = (c1.reshape(6, 14, 2, 14, 2).sum(axis=(2, 4))) >> 2

    # Conv2
    c2_in = im2col_fast(p1, 5, 5).astype(np.int64)
    c2_raw = (c2_in @ w_c2.T) >> shift
    c2 = np.clip(c2_raw + b_c2, 0, max_val).reshape(10, 10, 16).transpose(2, 0, 1)

    # Pool2
    p2 = (c2.reshape(16, 5, 2, 5, 2).sum(axis=(2, 4))) >> 2

    # Conv3
    c3_in = p2.flatten()[None, :].astype(np.int64)
    c3_raw = (c3_in @ w_c3.T) >> shift
    c3 = np.clip((c3_raw + b_c3).flatten(), 0, max_val)

    # FC1
    fc1_raw = (c3[None, :] @ w_f1.T) >> shift
    fc1 = np.clip((fc1_raw + b_f1).flatten(), 0, max_val)

    # FC2
    fc2_raw = (fc1[None, :] @ w_f2.T) >> shift
    fc2 = np.clip((fc2_raw + b_f2).flatten(), min_val, max_val)

    return np.argmax(fc2)

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wdir = os.path.join(root_dir, 'data', 'weights')
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

    data_dir = os.path.join(root_dir, 'data')
    images = read_hex_images(os.path.join(data_dir, 'test_images.hex')).reshape(20, 32, 32)
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
