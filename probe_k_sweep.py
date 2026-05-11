"""
Sweep k_sparse on Pythia 410M dense_4h_to_h to find the smallest k that
preserves the codebook win. Smaller k => smaller bytes-per-matrix.
"""
import modal

app = modal.App("probe-k-sweep")

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


@app.function(image=image, gpu="A10G", timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_name("huggingface")])
def run(layer: int = 12):
    import json, time, math
    import torch
    from transformers import AutoModelForCausalLM
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    device = "cuda"
    torch.set_grad_enabled(False)

    print(f"=== k_sparse sweep, Pythia 410M L={layer} dense_4h_to_h ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-410m", torch_dtype=torch.float32, device_map="cuda",
    )
    model.eval()
    d_model = model.config.hidden_size

    W = model.gpt_neox.layers[layer].mlp.dense_4h_to_h.weight.detach().clone().to(torch.float32)
    path = hf_hub_download(repo_id="EleutherAI/sae-pythia-410m-65k", filename=f"layers.{layer}.mlp/sae.safetensors")
    st = load_file(path)
    W_dec = st["W_dec"]
    if W_dec.shape[1] != d_model: W_dec = W_dec.T
    D = W_dec.to(device=device, dtype=torch.float32).T.contiguous()
    K = D.shape[1]
    g = torch.Generator(device=device).manual_seed(0)
    D_rand = torch.randn(d_model, K, device=device, generator=g, dtype=torch.float32)
    D_rand = D_rand / D_rand.norm(dim=0, keepdim=True).clamp(min=1e-8)

    n_in = W.shape[1]
    print(f"  W shape={tuple(W.shape)}  K={K}", flush=True)

    ks = [1, 2, 4, 8, 16, 32, 64]
    results = []
    for k in ks:
        t0 = time.time()
        Z_sae = batched_mp(D, W, k)
        relerr_sae = ((W - D @ Z_sae).norm() / W.norm()).item()
        Z_rand = batched_mp(D_rand, W, k)
        relerr_rand = ((W - D_rand @ Z_rand).norm() / W.norm()).item()
        index_bits = math.ceil(math.log2(K))
        value_bits = 8
        bytes_per_matrix = (k * n_in * (index_bits + value_bits) + 7) // 8 + n_in * 2  # +scales
        results.append({
            "k_sparse": k,
            "relerr_sae": relerr_sae,
            "relerr_rand": relerr_rand,
            "gain": relerr_rand - relerr_sae,
            "bytes_per_matrix": bytes_per_matrix,
        })
        print(f"  k={k:>3}  sae={relerr_sae:.4f}  rand={relerr_rand:.4f}  gain={relerr_rand-relerr_sae:+.4f}  "
              f"bytes={bytes_per_matrix:>8,}  ({time.time()-t0:.2f}s)", flush=True)

    print("\n=== summary ===", flush=True)
    print(f"{'k':>3} {'sae':>8} {'rand':>8} {'gain':>8} {'bytes':>10}", flush=True)
    for r in results:
        print(f"{r['k_sparse']:>3} {r['relerr_sae']:.4f}  {r['relerr_rand']:.4f}  {r['gain']:+.4f}  {r['bytes_per_matrix']:>10,}", flush=True)
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
def main(layer: int = 12):
    print(run.remote(layer=layer))
