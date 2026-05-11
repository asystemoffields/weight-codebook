"""Quick smoke test of kv_virtual_atoms_probe machinery (one layer, one head, one n, one m)."""
import sys, math, time
sys.path.insert(0, r"C:\Users\power\documents\weight-codebook")

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

import kv_virtual_atoms_probe as P

device = "cpu"
torch.set_grad_enabled(False)

tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
cfg = model.config
n_heads = cfg.n_head
head_dim = cfg.n_embd // n_heads
scale = 1.0 / math.sqrt(head_dim)

text = P.PASSAGES["prose"]
ids = tok(text, return_tensors="pt").input_ids.to(device)
print(f"seq_len = {ids.shape[1]}")

t0 = time.time()
capt = P.capture_qkv(model, ids)
print(f"capture_qkv: {time.time()-t0:.2f}s ({len(capt)} layers)")

layer_i, head_i = 5, 3
Q = capt[layer_i]["q"][head_i].float()
K = capt[layer_i]["k"][head_i].float()
V = capt[layer_i]["v"][head_i].float()
print(f"Q/K/V shapes (head): {Q.shape}, {K.shape}, {V.shape}")

n, m, r = 64, 8, 32
K_old, V_old = K[:n], V[:n]
Q_future = Q[n:n+r]
r_train = r // 2
Q_train, Q_test = Q_future[:r_train], Q_future[r_train:]

with torch.no_grad():
    Y_train_true = P.attn_output(Q_train, K_old, V_old, scale)
    Y_test_true  = P.attn_output(Q_test, K_old, V_old, scale)

# virtual atoms
t0 = time.time()
with torch.enable_grad():
    Pv, Uv, losses = P.fit_virtual_atoms(K_old, V_old, Q_train, Y_train_true, m,
                                          scale=scale, steps=300, lr=5e-2, seed=0)
with torch.no_grad():
    Yh_test = P.attn_output(Q_test, Pv, Uv, scale)
print(f"virtual_atoms fit: {time.time()-t0:.2f}s, final train loss={losses[-1]:.5f}, n_steps={len(losses)}")
print(f"  test rel_mse={P.rel_mse(Yh_test, Y_test_true):.5f}, cos={P.mean_cosine(Yh_test, Y_test_true):.5f}")

# baselines
Pr, Ur = P.random_subset_baseline(K_old, V_old, m, seed=0)
Pi, Ui = P.importance_subset_baseline(K_old, V_old, Q_train, m, scale)
Pk, Uk = P.kmeans_baseline(K_old, V_old, m, scale, seed=0)
for name, (Pm, Um) in [("random", (Pr, Ur)), ("importance", (Pi, Ui)), ("kmeans", (Pk, Uk))]:
    with torch.no_grad():
        Yh = P.attn_output(Q_test, Pm, Um, scale)
    print(f"  {name:11s}: rel_mse={P.rel_mse(Yh, Y_test_true):.5f}, cos={P.mean_cosine(Yh, Y_test_true):.5f}")

print("smoke OK")
