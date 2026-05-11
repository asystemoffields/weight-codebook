"""
Cross-layer SAE codebook probe — does ONE dictionary serve MANY layers?

If we need a separate ~256 MB SAE dictionary for every layer in Pythia 410M,
the dictionary cost (24 × 256 MB = 6 GB) overwhelms the model itself. The
compression-by-codebook idea only saves bytes if one dictionary amortizes
across many matrices.

This probe: pick a single source layer L_src in {4, 8, 12, 16, 20}. Use its
MLP-output SAE as the dictionary. Sparse-code dense_4h_to_h of EVERY OTHER
layer over that single dictionary. Measure relerr at each L_tgt.

If the relerr stays close to the same-layer-SAE relerr (~0.55-0.60), the
dictionary transfers and we have a real compression scheme. If it explodes,
each layer needs its own dictionary.
"""
import modal

app = modal.App("probe-cross-layer-pythia")

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

    print(f"=== Cross-layer codebook probe (Pythia 410M) ===", flush=True)
    print(f"  k_sparse={k_sparse}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-410m", torch_dtype=torch.float32, device_map="cuda",
    )
    model.eval()
    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f"  d_model={d_model}  n_layers={n_layers}", flush=True)

    # Pre-extract all W matrices
    Ws = {}
    for L in range(n_layers):
        Ws[L] = model.gpt_neox.layers[L].mlp.dense_4h_to_h.weight.detach().clone().to(torch.float32)

    sae_repo = "EleutherAI/sae-pythia-410m-65k"
    sae_cache = {}
    def get_D(L):
        if L in sae_cache: return sae_cache[L]
        t0 = time.time()
        path = hf_hub_download(repo_id=sae_repo, filename=f"layers.{L}.mlp/sae.safetensors")
        st = load_file(path)
        W_dec = st["W_dec"]
        if W_dec.shape[1] != d_model: W_dec = W_dec.T
        D = W_dec.to(device=device, dtype=torch.float32).T.contiguous()
        sae_cache[L] = D
        print(f"  loaded SAE L={L} ({time.time()-t0:.1f}s, D={D.shape})", flush=True)
        return D

    g_rand = torch.Generator(device=device).manual_seed(0)
    D_rand = None  # built lazily

    sources = [4, 8, 12, 16, 20]
    targets = list(range(0, n_layers, 2))    # layers 0, 2, 4, ..., 22

    results = []
    for L_src in sources:
        D = get_D(L_src)
        K = D.shape[1]
        if D_rand is None:
            D_rand = torch.randn(d_model, K, device=device, generator=g_rand, dtype=torch.float32)
            D_rand = D_rand / D_rand.norm(dim=0, keepdim=True).clamp(min=1e-8)
        print(f"\n--- Source SAE: L_src={L_src} (K={K}) ---", flush=True)
        for L_tgt in targets:
            W = Ws[L_tgt]
            Z = batched_mp(D, W, k_sparse)
            relerr_sae = ((W - D @ Z).norm() / W.norm()).item()
            Z_r = batched_mp(D_rand, W, k_sparse)
            relerr_rand = ((W - D_rand @ Z_r).norm() / W.norm()).item()
            results.append({"L_src": L_src, "L_tgt": L_tgt, "relerr_sae": relerr_sae, "relerr_rand": relerr_rand,
                            "gain": relerr_rand - relerr_sae})
            print(f"  L_src={L_src} -> L_tgt={L_tgt}:  sae={relerr_sae:.4f}  rand={relerr_rand:.4f}  gain={results[-1]['gain']:+.4f}", flush=True)

    # Summary: for each L_src, the mean gain across all targets
    print("\n=== CROSS-LAYER SUMMARY (rows: source SAE, cols: target W layer) ===", flush=True)
    header = "src\\tgt | " + " ".join(f"{L:>5}" for L in targets)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for L_src in sources:
        row = [r for r in results if r["L_src"] == L_src]
        cells = " ".join(f"{r['relerr_sae']:5.3f}" for r in row)
        mean_gain = sum(r['gain'] for r in row) / len(row)
        print(f"L={L_src:>2}    | {cells}   (mean gain {mean_gain:+.3f})", flush=True)

    # Also report: for each target layer, which source SAE works best
    print("\n=== Best source SAE per target ===", flush=True)
    for L_tgt in targets:
        best = min((r for r in results if r['L_tgt'] == L_tgt), key=lambda r: r['relerr_sae'])
        print(f"  L_tgt={L_tgt}: best source = L_src={best['L_src']}  relerr={best['relerr_sae']:.4f}", flush=True)

    print(json.dumps(results, indent=2), flush=True)
    return results


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
