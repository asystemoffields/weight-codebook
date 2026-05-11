"""
Multi-target codebook probe.

The same-layer / W_O / layer-5 probe was inconclusive (SAE dict was no better
than random gaussian at K=24576). Three structural questions are open:

  1. Is *any* layer's W_O sparse-codable over residual-stream SAE atoms?
  2. Is `mlp.c_proj` (which writes MLP output directly into the residual stream
     the SAE was trained on) the cleaner target than `attn.c_proj`?
  3. Does the test work better with the SAE from the layer AFTER the matrix
     (whose hook_resid_pre = residual stream the matrix has just written into)?

This probe sweeps (layer, weight_target, sae_offset) and reports relative
Frobenius reconstruction error against three baselines:
  - random gaussian dictionary at same K (lower bound from compressed sensing)
  - SVD at matched bytes
  - same-layer top-k SVD (best linear approximation at same rank)

The fingerprint we want: codebook relerr << random_dict relerr. If that gap
doesn't appear anywhere, the hypothesis is genuinely dead for GPT-2.
"""
import modal

app = modal.App("weight-codebook-multi")

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
def probe_multi(k_sparse: int = 32):
    import os, math, time, json
    import torch, numpy as np
    from transformers import AutoModelForCausalLM
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    device = "cuda"
    torch.set_grad_enabled(False)

    print(f"=== Multi-target codebook probe ===", flush=True)
    print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32, device_map="cuda")
    model.eval()
    cfg = model.config
    d_model = cfg.n_embd

    sae_repo = "jbloom/GPT2-Small-SAEs-Reformatted"
    sae_cache = {}

    def get_D(layer_for_sae: int):
        """Return (d_model, K) dictionary D from blocks.{layer_for_sae}.hook_resid_pre."""
        if layer_for_sae in sae_cache:
            return sae_cache[layer_for_sae]
        folder = f"blocks.{layer_for_sae}.hook_resid_pre"
        path = hf_hub_download(repo_id=sae_repo, filename=f"{folder}/sae_weights.safetensors")
        st = load_file(path)
        W_dec = st["W_dec"]  # (K, d_model)
        if W_dec.shape[1] != d_model:
            W_dec = W_dec.T
        D = W_dec.to(device=device, dtype=torch.float32).T.contiguous()  # (d_model, K)
        sae_cache[layer_for_sae] = D
        print(f"    loaded SAE for layer_for_sae={layer_for_sae}: D shape {tuple(D.shape)}", flush=True)
        return D

    results = []

    # Test on every other layer plus key positions
    target_layers = [1, 3, 5, 8, 10]
    weight_targets = ["attn.c_proj", "mlp.c_proj"]
    sae_offsets = [0, 1]  # SAE from same layer's pre, or next layer's pre

    g_rand = torch.Generator(device=device).manual_seed(0)

    for L in target_layers:
        block = model.transformer.h[L]
        for wt in weight_targets:
            if wt == "attn.c_proj":
                W = block.attn.c_proj.weight.detach().clone().to(torch.float32)
            else:
                W = block.mlp.c_proj.weight.detach().clone().to(torch.float32)
            # GPT-2 Conv1D weight is (in, out). For sparse-coding the contribution
            # to residual stream, we treat columns of W (each column = one input
            # dim's contribution direction in residual stream). For attn.c_proj
            # shape is (d_model, d_model); for mlp.c_proj it's (4*d_model, d_model).
            # In Conv1D y = x @ W; "the output direction produced by input dim i" is
            # W[i, :], i.e. ROWS of W are vectors in d_model space. So we should
            # sparse-code ROWS (or equivalently columns of W.T).
            #
            # Rewriting target as Y where each COLUMN of Y is a vector in R^d_model:
            Y = W.T.contiguous()  # (d_model, n_inputs)
            d_out, n_in = Y.shape
            assert d_out == d_model, f"unexpected Y shape {Y.shape}"

            for sae_off in sae_offsets:
                L_sae = L + sae_off
                if L_sae >= cfg.n_layer:
                    continue
                D = get_D(L_sae)
                K = D.shape[1]

                Z = batched_matching_pursuit(D, Y, k_sparse)
                Y_cb = D @ Z
                relerr_cb = ((Y - Y_cb).norm() / Y.norm()).item()

                # Random gaussian dictionary at same K (compressed sensing baseline)
                D_rand = torch.randn(d_model, K, device=device, generator=g_rand, dtype=torch.float32)
                D_rand = D_rand / D_rand.norm(dim=0, keepdim=True).clamp(min=1e-8)
                Z_r = batched_matching_pursuit(D_rand, Y, k_sparse)
                Y_rand = D_rand @ Z_r
                relerr_rand = ((Y - Y_rand).norm() / Y.norm()).item()

                # SVD at matched bytes
                codebook_bytes = 4 * k_sparse * n_in
                rank_match = max(1, codebook_bytes // (2 * (d_model + n_in)))
                U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
                Y_svd = (U[:, :rank_match] * S[:rank_match]) @ Vh[:rank_match, :]
                relerr_svd = ((Y - Y_svd).norm() / Y.norm()).item()

                # Best-possible at same rank (k_sparse) -- top-k_sparse SVD
                # gives a lower bound on what any rank-k_sparse linear basis can do.
                Y_svdK = (U[:, :k_sparse] * S[:k_sparse]) @ Vh[:k_sparse, :]
                relerr_svdK = ((Y - Y_svdK).norm() / Y.norm()).item()

                rec = {
                    "L": L, "weight": wt, "L_sae": L_sae, "K": K, "n_in": n_in,
                    "k_sparse": k_sparse, "rank_match": rank_match,
                    "relerr_codebook": relerr_cb,
                    "relerr_random_dict": relerr_rand,
                    "relerr_svd_matched": relerr_svd,
                    "relerr_svd_rankK": relerr_svdK,
                    "gain_vs_random": relerr_rand - relerr_cb,
                }
                results.append(rec)
                print(f"  L={L} {wt:13s} sae=L{L_sae}  cb={relerr_cb:.4f}  rand={relerr_rand:.4f}  "
                      f"svd_match={relerr_svd:.4f}  svd_rankK={relerr_svdK:.4f}  "
                      f"gain={rec['gain_vs_random']:+.4f}", flush=True)

    # Summarize
    print("\n=== SUMMARY ===", flush=True)
    wins = [r for r in results if r["gain_vs_random"] > 0.05]
    ties = [r for r in results if -0.02 <= r["gain_vs_random"] <= 0.05]
    losses = [r for r in results if r["gain_vs_random"] < -0.02]
    print(f"  configs where codebook beats random by >5pp: {len(wins)}/{len(results)}", flush=True)
    print(f"  configs near tie (-2 to +5 pp): {len(ties)}/{len(results)}", flush=True)
    print(f"  configs where codebook LOSES to random by >2pp: {len(losses)}/{len(results)}", flush=True)
    if wins:
        print("  wins:", flush=True)
        for r in sorted(wins, key=lambda r: -r["gain_vs_random"]):
            print(f"    L={r['L']} {r['weight']} sae=L{r['L_sae']}: gain={r['gain_vs_random']:+.4f}", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    return results


def batched_matching_pursuit(D, Y, k):
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
    res = probe_multi.remote(k_sparse=k_sparse)
    print(res)
