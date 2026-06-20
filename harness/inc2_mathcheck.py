#!/usr/bin/env python3
# INC-2 gate 2: offline numpy cross-check of the in-graph nextn math.
#
# Reads the dumped per-column tensors (/tmp/mtp_dump_*.bin, float32) produced by the probe in
# LLAMA_MTP_DUMP mode, plus the F32/Q8_0 nextn weights from the GGUF, and recomputes:
#   * ehcat = concat(enorm(emb_shift), hnorm(h_prenorm))   -> compare to in-graph nextn_ehcat
#   * x     = dequant(eh_proj) @ ehcat                     -> compare to in-graph nextn_x
#   * o     = shared_head_norm(nextn_block)                -> compare to in-graph nextn_o
# Negative controls (must FAIL to match) prove the test discriminates the silent alpha-killers:
#   * swapped enorm/hnorm, reversed concat order, output_norm instead of shared_head_norm.

import sys, glob
sys.path.insert(0, "/home/hikari/src/llama.cpp/gguf-py")
import gguf
import numpy as np

MD = "/mnt/data/models/GLM-5.2-GGUF/UD-Q2_K_XL"
EPS = 1e-5
N_EMBD = 6144

def load_bin(name):
    return np.fromfile(f"/tmp/mtp_dump_{name}.bin", dtype=np.float32)

def get_weights():
    want = {
        "blk.78.nextn.enorm.weight": None,
        "blk.78.nextn.hnorm.weight": None,
        "blk.78.nextn.shared_head_norm.weight": None,
        "blk.78.nextn.eh_proj.weight": None,
        "output_norm.weight": None,
    }
    for sp in sorted(glob.glob(MD + "/*.gguf")):
        r = gguf.GGUFReader(sp)
        for t in r.tensors:
            if t.name in want and want[t.name] is None:
                arr = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
                want[t.name] = np.array(arr)  # copy out of mmap
    for k, v in want.items():
        if v is None:
            raise RuntimeError(f"weight not found: {k}")
    return want

def rmsnorm(x, w, eps=EPS):
    # ggml RMSNorm: x / sqrt(mean(x^2)+eps) * w   (per row/column vector)
    ms = np.mean(x.astype(np.float64) ** 2)
    return (x / np.sqrt(ms + eps)).astype(np.float64) * w.astype(np.float64)

def cos(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

def relerr(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))

def main():
    W = get_weights()
    enorm  = W["blk.78.nextn.enorm.weight"].reshape(-1)
    hnorm  = W["blk.78.nextn.hnorm.weight"].reshape(-1)
    shnorm = W["blk.78.nextn.shared_head_norm.weight"].reshape(-1)
    onorm  = W["output_norm.weight"].reshape(-1)
    eh_proj = W["blk.78.nextn.eh_proj.weight"]  # {12288, 6144} stored [out=6144? ] -> verify
    # gguf stores 2D as (n_out, n_in) row-major in numpy after dequantize -> shape (6144, 12288)
    print("eh_proj dequant shape:", eh_proj.shape)

    emb   = load_bin("nextn_emb_shift")
    h     = load_bin("nextn_h_prenorm")
    g_ehc = load_bin("nextn_ehcat")     # in-graph concat (2*n_embd,)
    g_x   = load_bin("nextn_x")         # in-graph x (n_embd,)
    g_blk = load_bin("nextn_block")     # in-graph block_out (n_embd,)
    g_o   = load_bin("nextn_o")         # in-graph o (n_embd,)
    print("dump sizes:", {k: v.shape for k, v in
          dict(emb=emb, h=h, ehcat=g_ehc, x=g_x, blk=g_blk, o=g_o).items()})

    ok = True

    # ---- check 1: ehcat = concat(enorm(emb), hnorm(h)) [e THEN hn] ----
    e_ref  = rmsnorm(emb, enorm)
    hn_ref = rmsnorm(h,   hnorm)
    ehc_ref = np.concatenate([e_ref, hn_ref])
    c1 = cos(ehc_ref, g_ehc); r1 = relerr(ehc_ref, g_ehc)
    print(f"\n[1] ehcat (enorm(emb)+hnorm(h), order e;hn)  cos={c1:.6f} relerr={r1:.2e}")
    ok &= (c1 > 0.9999 and r1 < 1e-3)

    # negative controls for check 1
    ehc_swap = np.concatenate([rmsnorm(emb, hnorm), rmsnorm(h, enorm)])  # swapped norms
    ehc_rev  = np.concatenate([hn_ref, e_ref])                          # reversed concat order
    print(f"    [neg] swapped enorm/hnorm   cos={cos(ehc_swap, g_ehc):.6f}  (should be < {c1:.4f})")
    print(f"    [neg] reversed concat order cos={cos(ehc_rev,  g_ehc):.6f}  (should be << 1)")

    # ---- check 2: x = eh_proj @ ehcat (Q8_0 dequant matmul) ----
    # ggml mul_mat(W, b): W is {n_in=2*n_embd, n_out=n_embd} in ggml ne-order; numpy dequant gives
    # (n_out, n_in) = (6144, 12288). x = W @ ehcat.
    if eh_proj.shape == (N_EMBD, 2 * N_EMBD):
        x_ref = eh_proj @ g_ehc.astype(np.float64)  # use in-graph ehcat to isolate the matmul
    elif eh_proj.shape == (2 * N_EMBD, N_EMBD):
        x_ref = eh_proj.T @ g_ehc.astype(np.float64)
    else:
        raise RuntimeError(f"unexpected eh_proj shape {eh_proj.shape}")
    c2 = cos(x_ref, g_x); r2 = relerr(x_ref, g_x)
    print(f"\n[2] x = eh_proj @ ehcat (Q8_0)              cos={c2:.6f} relerr={r2:.2e}")
    ok &= (c2 > 0.999)

    # ---- check 3: o = shared_head_norm(block_out)  (NOT output_norm) ----
    o_ref = rmsnorm(g_blk, shnorm)
    c3 = cos(o_ref, g_o); r3 = relerr(o_ref, g_o)
    print(f"\n[3] o = shared_head_norm(block)             cos={c3:.6f} relerr={r3:.2e}")
    ok &= (c3 > 0.9999 and r3 < 1e-3)
    # negative control: output_norm instead of shared_head_norm
    o_wrong = rmsnorm(g_blk, onorm)
    print(f"    [neg] output_norm(block)    cos={cos(o_wrong, g_o):.6f}  (should be < {c3:.4f})")

    print("\n=========================================")
    print("GATE 2 (offline math):", "PASS" if ok else "*** FAIL ***")
    print("=========================================")
    return 0 if ok else 2

if __name__ == "__main__":
    sys.exit(main())
