"""
Q/K/V projection codebook probe.

We've confirmed dense_4h_to_h (MLP down) is sparse-codable over MLP-output SAE.
But MLP is only ~50% of model bytes. Attention projections (Q, K, V, O) are
the other half. Do they live in span(D) too?

In Pythia, the attention matrices are packed: query_key_value is one tensor of
shape (3 * d_model, d_model). It maps residual stream -> concat(Q, K, V), then
attention does its thing, then `dense` projects back into residual stream.

For each layer L we test 4 matrices:
  - W_Q (top d_model rows of query_key_value.weight)
  - W_K (middle d_model rows)
  - W_V (bottom d_model rows)
  - W_O (dense.weight)

Two dictionaries to compare:
  - The MLP-output SAE at the SAME layer (might span Q/K/V outputs if those
    live in similar feature space as MLP outputs).
  - Random gaussian at same K.

Outputs that live in residual stream space: only W_O writes there. W_Q/K/V
write into attention internal space, which is NOT what the MLP SAE was trained
on. So we expect:
  - W_O: should show codebook win (similar to MLP results).
  - W_Q/K/V: probably no win, but we measure to be sure.

If even W_Q/K/V show a partial structural signal, the dictionary is broader
than just "spans residual stream writes."
"""
import modal

app = modal.App("probe-qkv-codebook")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2.0", "torch==2.5.1", "transformers==4.46.3",
        "accelerate>=1.0.0", "safetensors>=0.4.5,<1.0",
        "huggingface_hub[hf_transfer]>=0.26,<1.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
)
hf_cache = modal.Volume.from_name("hf-cache-weight-codebook", create_if_missing=True)


@app.function(
    image=image, gpu="A10G", timeout=3600,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface")],
)
def run(k_sparse: int = 32):
    import os, json, time
    import torch, numpy as np
    from transformers import AutoModelForCausalLM
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    device = "cuda"
    torch.set_grad_enabled(False)

    print(f"=== Q/K/V codebook probe (Pythia 410M) ===  k_sparse={k_sparse}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-410m", torch_dtype=torch.float32, device_map="cuda",
    )
    model.eval()
    d_model = model.config.hidden_size

    def get_D(L):
        path = hf_hub_download(repo_id="EleutherAI/sae-pythia-410m-65k", filename=f"layers.{L}.mlp/sae.safetensors")
        st = load_file(path)
        W_dec = st["W_dec"]
        if W_dec.shape[1] != d_model: W_dec = W_dec.T
        return W_dec.to(device=device, dtype=torch.float32).T.contiguous()

    target_layers = [4, 12, 20]
    g_rand = torch.Generator(device=device).manual_seed(0)
    K = 65536
    D_rand = torch.randn(d_model, K, device=device, generator=g_rand, dtype=torch.float32)
    D_rand = D_rand / D_rand.norm(dim=0, keepdim=True).clamp(min=1e-8)

    rows = []
    for L in target_layers:
        print(f"\n--- layer {L} ---", flush=True)
        block = model.gpt_neox.layers[L]
        D = get_D(L)

        # query_key_value: shape (3*d_model, d_model). split into Q,K,V each (d_model, d_model)
        Wqkv = block.attention.query_key_value.weight.detach().clone().to(torch.float32)
        Wq, Wk, Wv = Wqkv.split(d_model, dim=0)
        # dense: shape (d_model, d_model) — the W_O analog
        Wo = block.attention.dense.weight.detach().clone().to(torch.float32)

        matrices = {"W_Q": Wq, "W_K": Wk, "W_V": Wv, "W_O": Wo}

        for name, W in matrices.items():
            # All shape (d_model, d_model). Treat rows as output directions.
            # For W_O, rows are residual-stream output directions per attn-input dim.
            # For W_Q/K/V, rows are attn-space directions per residual-stream input dim.
            # In both cases, target Y has shape (d_model, n_in).
            Y = W   # (d_model, d_model)
            Z = batched_mp(D, Y, k_sparse)
            relerr_sae = ((Y - D @ Z).norm() / Y.norm()).item()
            Z_r = batched_mp(D_rand, Y, k_sparse)
            relerr_rand = ((Y - D_rand @ Z_r).norm() / Y.norm()).item()
            gain = relerr_rand - relerr_sae
            print(f"  L={L:>2} {name}: sae={relerr_sae:.4f}  rand={relerr_rand:.4f}  gain={gain:+.4f}", flush=True)
            rows.append({"L": L, "matrix": name, "relerr_sae": relerr_sae, "relerr_rand": relerr_rand, "gain": gain})

    print("\n=== SUMMARY ===", flush=True)
    print(f"{'L':>3} {'matrix':>6} {'sae':>8} {'rand':>8} {'gain':>8}", flush=True)
    for r in rows:
        print(f"{r['L']:>3} {r['matrix']:>6} {r['relerr_sae']:.4f}  {r['relerr_rand']:.4f}  {r['gain']:+.4f}", flush=True)
    wins = sum(1 for r in rows if r["gain"] > 0.05)
    print(f"\nconfigs with codebook win >5pp: {wins}/{len(rows)}", flush=True)
    print(json.dumps(rows, indent=2), flush=True)
    return rows


def batched_mp(D, Y, k):
    import torch
    d, K = D.shape; _, n = Y.shape; device = Y.device
    dtype = torch.float32
    D = D.to(dtype); Y = Y.to(dtype)
    D_norms = D.norm(dim=0).clamp(min=1e-8)
    D_unit = D / D_norms.unsqueeze(0)
    residual = Y.clone()
    Z = torch.zeros(K, n, device=device, dtype=dtype)
    col_idx = torch.arange(n, device=device)
    for step in range(k):
        corr = D_unit.T @ residual
        atom = corr.abs().argmax(dim=0)
        coeff_unit = corr[atom, col_idx]
        residual = residual - D_unit[:, atom] * coeff_unit.unsqueeze(0)
        Z[atom, col_idx] = Z[atom, col_idx] + coeff_unit / D_norms[atom]
    return Z


@app.local_entrypoint()
def main(k_sparse: int = 32):
    res = run.remote(k_sparse=k_sparse)
    print(res)
