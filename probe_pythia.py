"""
Pythia 410M codebook probe — bigger non-toy model validation.

The GPT-2 small probes confirmed:
  * SAE codebook beats random gaussian dict at the same K (12/20 configs)
  * SAE codebook beats activation-PCA at matched effective K (5/5)
  * Best when matching W's output space to the NEXT layer's residual SAE

This probe replicates the test on Pythia 410M (d_model=1024, 24 layers, 3.3x
bigger than GPT-2). EleutherAI's sae-pythia-410m-65k provides MLP-OUTPUT SAEs
at every layer — directly the space `dense_4h_to_h` writes into.

If codebook beats random here too, the structural fingerprint generalizes
beyond GPT-2 small.
"""
import modal

app = modal.App("probe-pythia-codebook")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2.0",
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate>=1.0.0",
        "safetensors>=0.4.5,<1.0",
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

    print(f"=== Pythia 410M codebook probe ===", flush=True)
    print(f"  k_sparse={k_sparse}  GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print("\n[1] loading pythia-410m...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-410m",
        torch_dtype=torch.float32, device_map="cuda",
    )
    model.eval()
    cfg = model.config
    d_model = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
    print(f"    loaded in {time.time()-t0:.1f}s; d_model={d_model} n_layers={n_layers}", flush=True)

    sae_repo = "EleutherAI/sae-pythia-410m-65k"
    sae_cache = {}

    def get_D(layer: int):
        if layer in sae_cache: return sae_cache[layer]
        t0 = time.time()
        path = hf_hub_download(repo_id=sae_repo, filename=f"layers.{layer}.mlp/sae.safetensors")
        st = load_file(path)
        print(f"    SAE layer {layer} keys: {list(st.keys())}  load {time.time()-t0:.1f}s", flush=True)
        # Try standard sparsify key naming
        if "W_dec" in st:
            W_dec = st["W_dec"]
        elif "decoder.weight" in st:
            # sparsify lib: decoder is nn.Linear(K, d_model), so weight shape (d_model, K)
            W_dec = st["decoder.weight"].T   # transpose to (K, d_model)
        else:
            cand = [k for k in st.keys() if "dec" in k.lower() and st[k].ndim == 2]
            if not cand:
                raise RuntimeError(f"No decoder weight in SAE; keys: {list(st.keys())}")
            W_dec = st[cand[0]]
            if W_dec.shape[1] != d_model: W_dec = W_dec.T
        if W_dec.shape[1] != d_model:
            W_dec = W_dec.T
        D = W_dec.to(device=device, dtype=torch.float32).T.contiguous()  # (d_model, K)
        sae_cache[layer] = D
        return D

    target_layers = [4, 8, 12, 16, 20]   # spread across 24 layers
    g_rand = torch.Generator(device=device).manual_seed(0)
    results = []

    for L in target_layers:
        # Pythia MLP down-projection
        block = model.gpt_neox.layers[L]
        W = block.mlp.dense_4h_to_h.weight.detach().clone().to(torch.float32)
        # nn.Linear: weight shape (out, in) = (d_model, 4*d_model). Y has columns = output
        # direction per input neuron, so Y = W (already in that orientation).
        Y = W
        print(f"\n[L={L}] dense_4h_to_h shape: {tuple(Y.shape)}", flush=True)

        # SAE for THIS layer's mlp output (directly the space Y writes into)
        D = get_D(L)
        K = D.shape[1]

        # Sparse coding
        t0 = time.time()
        Z = batched_mp(D, Y, k_sparse)
        Y_hat = D @ Z
        relerr_sae = ((Y - Y_hat).norm() / Y.norm()).item()
        print(f"    SAE codebook (K={K}, k={k_sparse}): relerr={relerr_sae:.4f}  ({time.time()-t0:.1f}s)", flush=True)

        # Random gaussian dict at same K
        D_rand = torch.randn(d_model, K, device=device, generator=g_rand, dtype=torch.float32)
        D_rand = D_rand / D_rand.norm(dim=0, keepdim=True).clamp(min=1e-8)
        Z_r = batched_mp(D_rand, Y, k_sparse)
        Y_rand = D_rand @ Z_r
        relerr_rand = ((Y - Y_rand).norm() / Y.norm()).item()
        print(f"    Random dict (K={K}): relerr={relerr_rand:.4f}", flush=True)

        # SVD rank-k
        U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
        Y_svdK = (U[:, :k_sparse] * S[:k_sparse]) @ Vh[:k_sparse, :]
        relerr_svdK = ((Y - Y_svdK).norm() / Y.norm()).item()
        print(f"    SVD rank-{k_sparse}: relerr={relerr_svdK:.4f}", flush=True)

        rec = {
            "L": L, "d_model": d_model, "K": K, "k_sparse": k_sparse,
            "relerr_sae": relerr_sae, "relerr_rand": relerr_rand, "relerr_svdK": relerr_svdK,
            "gain_vs_random": relerr_rand - relerr_sae,
        }
        results.append(rec)

    print("\n=== SUMMARY (Pythia 410M dense_4h_to_h) ===", flush=True)
    print(f"{'L':>3} {'SAE':>8} {'Rand':>8} {'SVD-K':>8} {'gain':>8}", flush=True)
    for r in results:
        print(f"{r['L']:>3}  {r['relerr_sae']:.4f}  {r['relerr_rand']:.4f}  {r['relerr_svdK']:.4f}  {r['gain_vs_random']:+.4f}", flush=True)
    wins = sum(1 for r in results if r['gain_vs_random'] > 0.05)
    print(f"\nconfigs where codebook beats random by >5pp: {wins}/{len(results)}", flush=True)
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
