"""
Full LeNet-5 Bit-Exact Simulator with FIXED Controller
======================================================
Tests all 20 FashionMNIST test images using the fixed Torus timing.
"""
import numpy as np
import os

def read_hex_weights(filename, count):
    vals = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                v = int(line, 16)
                if v >= 32768: v -= 65536
                vals.append(v)
    return np.array(vals, dtype=np.int64)

def read_hex_images(filename, count):
    vals = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                v = int(line, 16)
                if v >= 32768: v -= 65536
                vals.append(v)
    return np.array(vals, dtype=np.int64)

def asr_32bit(val, shift):
    val = val & 0xFFFFFFFF
    if val >= 0x80000000:
        val -= 0x100000000
    return val >> shift

def clamp_act(x):
    return np.clip(x, -256, 256)

def sat16(x):
    return np.clip(x, -32768, 32767)

def torus_tile_FIXED(a_tile, b_tile, mat_size=5, frac_bits=8):
    left_init = np.zeros((mat_size, mat_size), dtype=np.int64)
    up_init   = np.zeros((mat_size, mat_size), dtype=np.int64)
    for r in range(mat_size):
        for c in range(mat_size):
            temp = (mat_size - ((r + 1 + c) % mat_size)) % mat_size
            left_init[r][c] = b_tile[temp][r]
            up_init[r][c]   = a_tile[c][temp]
    
    up_reg   = up_init.copy()
    left_reg = left_init.copy()
    psum     = np.zeros((mat_size, mat_size), dtype=np.int64)
    
    pe_down  = np.zeros((mat_size, mat_size), dtype=np.int64)
    pe_right = np.zeros((mat_size, mat_size), dtype=np.int64)
    
    # Cycle 1: S_PRIME (MULT_ADD only)
    for r in range(mat_size):
        for c in range(mat_size):
            product = int(left_reg[r][c]) * int(up_reg[r][c])
            shifted = asr_32bit(product, frac_bits)
            psum[r][c] += shifted
    
    pe_down  = up_reg.copy()
    pe_right = left_reg.copy()
    
    # Cycles 2..5: S_RUN (MOVE + MULT_ADD)
    for step in range(mat_size - 1):
        new_up_reg   = np.zeros_like(up_reg)
        new_left_reg = np.zeros_like(left_reg)
        for r in range(mat_size):
            for c in range(mat_size):
                src_r = (r - 1) % mat_size
                new_up_reg[r][c] = pe_down[src_r][c]
                src_c = (c - 1) % mat_size
                new_left_reg[r][c] = pe_right[r][src_c]
        
        up_reg   = new_up_reg
        left_reg = new_left_reg
        
        for r in range(mat_size):
            for c in range(mat_size):
                product = int(left_reg[r][c]) * int(up_reg[r][c])
                shifted = asr_32bit(product, frac_bits)
                psum[r][c] += shifted
        
        pe_down  = up_reg.copy()
        pe_right = left_reg.copy()
    
    return psum

def run_gemm_rtl_sim(layer, N, K, M, get_a_fn, weights, bias):
    out = np.zeros((M, N), dtype=np.int64)
    SA_SIZE = 5
    for n_base in range(0, N, SA_SIZE):
        for m_base in range(0, M, SA_SIZE):
            acc_tile = np.zeros((SA_SIZE, SA_SIZE), dtype=np.int64)
            for k_base in range(0, K, SA_SIZE):
                a_tile = np.zeros((SA_SIZE, SA_SIZE), dtype=np.int64)
                for rr in range(SA_SIZE):
                    for cc in range(SA_SIZE):
                        n_idx = n_base + rr
                        k_idx = k_base + cc
                        if n_idx < N and k_idx < K:
                            a_tile[rr][cc] = get_a_fn(n_idx, k_idx)
                
                b_tile = np.zeros((SA_SIZE, SA_SIZE), dtype=np.int64)
                for rr in range(SA_SIZE):
                    for cc in range(SA_SIZE):
                        k_idx = k_base + rr
                        m_idx = m_base + cc
                        if k_idx < K and m_idx < M:
                            b_tile[rr][cc] = weights[m_idx * K + k_idx]
                
                p_tile = torus_tile_FIXED(a_tile, b_tile)
                acc_tile += p_tile
            
            for rr in range(SA_SIZE):
                for cc in range(SA_SIZE):
                    m_idx = m_base + rr
                    n_idx = n_base + cc
                    if n_idx < N and m_idx < M:
                        tile_sum = acc_tile[rr][cc] + bias[m_idx]
                        if layer != 6:
                            tile_sum = clamp_act(tile_sum)
                        else:
                            tile_sum = sat16(tile_sum)
                        out[m_idx, n_idx] = tile_sum
    return out

def run_pool(in_data, CH, H, W):
    out_h, out_w = H // 2, W // 2
    out = np.zeros((CH, out_h, out_w), dtype=np.int64)
    for ch in range(CH):
        for r in range(out_h):
            for c in range(out_w):
                s = (in_data[ch, 2*r, 2*c] +
                     in_data[ch, 2*r, 2*c+1] +
                     in_data[ch, 2*r+1, 2*c] +
                     in_data[ch, 2*r+1, 2*c+1])
                out[ch, r, c] = clamp_act(s >> 2)
    return out

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    wdir = os.path.join(base_dir, 'weights')
    w_c1 = read_hex_weights(os.path.join(wdir, 'conv1_weight.txt'), 6*25)
    b_c1 = read_hex_weights(os.path.join(wdir, 'conv1_bias.txt'), 6)
    w_c2 = read_hex_weights(os.path.join(wdir, 'conv2_weight.txt'), 16*150)
    b_c2 = read_hex_weights(os.path.join(wdir, 'conv2_bias.txt'), 16)
    w_c3 = read_hex_weights(os.path.join(wdir, 'conv3_weight.txt'), 120*400)
    b_c3 = read_hex_weights(os.path.join(wdir, 'conv3_bias.txt'), 120)
    w_f1 = read_hex_weights(os.path.join(wdir, 'fc1_weight.txt'), 84*120)
    b_f1 = read_hex_weights(os.path.join(wdir, 'fc1_bias.txt'), 84)
    w_f2 = read_hex_weights(os.path.join(wdir, 'fc2_weight.txt'), 10*84)
    b_f2 = read_hex_weights(os.path.join(wdir, 'fc2_bias.txt'), 10)

    images = read_hex_images(os.path.join(base_dir, 'verilog', 'test_data', 'test_images.hex'), 20*32*32).reshape(20, 32, 32)
    # Ground truth labels for first 20 test images of FashionMNIST
    labels = [9, 2, 1, 1, 6, 1, 4, 6, 5, 7, 4, 5, 7, 3, 4, 1, 2, 4, 8, 0]

    correct = 0
    print("="*60)
    print("SIMULATING FULL INFERENCE WITH FIXED CONTROLLER TIMING")
    print("="*60)

    for i in range(20):
        img = images[i]
        
        # Conv1
        def get_a_c1(n_idx, k_idx):
            oy, ox = n_idx // 28, n_idx % 28
            ky, kx = k_idx // 5, k_idx % 5
            return img[oy + ky, ox + kx]
        
        c1 = run_gemm_rtl_sim(0, 784, 25, 6, get_a_c1, w_c1, b_c1).reshape(6, 28, 28)
        p1 = run_pool(c1, 6, 28, 28)
        
        # Conv2
        p1_flat = p1.reshape(6, 14, 14)
        def get_a_c2(n_idx, k_idx):
            oy, ox = n_idx // 10, n_idx % 10
            ic = k_idx // 25
            rem = k_idx % 25
            ky, kx = rem // 5, rem % 5
            return p1_flat[ic, oy + ky, ox + kx]
        
        c2 = run_gemm_rtl_sim(2, 100, 150, 16, get_a_c2, w_c2, b_c2).reshape(16, 10, 10)
        p2 = run_pool(c2, 16, 10, 10)
        
        # Conv3
        p2_flat = p2.reshape(16, 5, 5)
        def get_a_c3(n_idx, k_idx):
            ic = k_idx // 25
            rem = k_idx % 25
            ky, kx = rem // 5, rem % 5
            return p2_flat[ic, ky, kx]
        
        c3 = run_gemm_rtl_sim(4, 1, 400, 120, get_a_c3, w_c3, b_c3).reshape(120)
        
        # FC1
        def get_a_fc1(n_idx, k_idx):
            return c3[k_idx]
        
        fc1 = run_gemm_rtl_sim(5, 1, 120, 84, get_a_fc1, w_f1, b_f1).reshape(84)
        
        # FC2
        def get_a_fc2(n_idx, k_idx):
            return fc1[k_idx]
        
        fc2 = run_gemm_rtl_sim(6, 1, 84, 10, get_a_fc2, w_f2, b_f2).reshape(10)
        
        pred = np.argmax(fc2)
        is_ok = (pred == labels[i])
        if is_ok: correct += 1
        
        status = "OK  " if is_ok else "WRONG"
        print(f"  img[{i:2d}] label={labels[i]} pred={pred} {status}  logits={fc2}")

    print("="*60)
    print(f"FINAL ACCURACY: {correct} / 20 = {100.0 * correct / 20:.1f}%")
    print("="*60)

if __name__ == '__main__':
    main()
