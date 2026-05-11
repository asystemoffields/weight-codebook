"""
Activation-PCA vs SAE-codebook probe.

If `probe_multi` confirms mlp.c_proj is sparse-codable over the next-layer SAE
atoms, the natural next question is: are SAE atoms doing anything beyond what
activation PCA does? That is, do they capture *more* than just the dominant
variance directions of the residual stream?

Procedure for each (target layer L, weight = mlp.c_proj):
  1. Run GPT-2 on a chunk of wikitext, capture residual stream activations at
     hook_resid_post of layer L (= hook_resid_pre of layer L+1).
  2. PCA: top-K principal directions of activations -> dictionary D_pca shape
     (d_model, K).
  3. Compare codebook reconstruction of W = mlp.c_proj using:
        - D_sae: SAE decoder atoms from layer L+1's hook_resid_pre
        - D_pca: top-K principal directions of activations
        - D_rand: random gaussian at same K (lower bound)
        - SVD rank-k_sparse baseline (best possible at same rank)
  4. Report: reconstruction relerr and PPL delta after substitution.

Verdict logic:
  - SAE > PCA: structural signal beyond variance; suggests SAE is the right basis.
  - PCA >= SAE: activation-aware low-rank is sufficient; no SAE needed.
  - Random >= both: hypothesis fully dead.
"""
import modal

app = modal.App("probe-activation-pca")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2.0",
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate>=1.0.0",
        "datasets>=3.0.0,<4.0.0",
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
def run(k_sparse: int = 32, n_activation_tokens: int = 16384, n_eval_tokens: int = 16384):
    import os, json, time
    import torch, numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from datasets import load_dataset

    device = "cuda"
    torch.set_grad_enabled(False)

    print(f"=== Activation-PCA vs SAE-codebook probe ===", flush=True)
    print(f"  k_sparse={k_sparse}  n_activation_tokens={n_activation_tokens}  n_eval_tokens={n_eval_tokens}", flush=True)

    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32, device_map="cuda")
    model.eval()
    tok = AutoTokenizer.from_pretrained("gpt2")
    cfg = model.config
    d_model = cfg.n_embd
    n_layers = cfg.n_layer

    # ---- collect activations at hook_resid_pre for each layer (+ final post) ----
    print(f"\n[1] collecting activations (n={n_activation_tokens})...", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in ds["text"] if t.strip())[:200_000]
    ids = tok(text, return_tensors="pt").input_ids[0, :n_activation_tokens].to(device)
    print(f"    using {ids.numel()} tokens", flush=True)

    # Capture pre-residual at each layer block
    capt = [[] for _ in range(n_layers + 1)]  # +1 for "after last block"
    handles = []
    def make_pre_hook(i):
        # hook on h[i] forward: input is hidden_states pre-block (=residual_stream entering block i)
        def hk(module, inputs, output):
            # output is a tuple (hidden_states_after, ...)
            pass
        return hk
    # Easier: capture via forward hooks on each block; record input (pre) and output (post)
    pre_holder = [None] * n_layers
    post_holder = [None] * n_layers
    def make_block_hook(i):
        def hk(module, inputs, output):
            # inputs[0] is hidden_states pre-block, output[0] is post-block hidden
            pre_holder[i] = inputs[0].detach()
            post_holder[i] = output[0].detach() if isinstance(output, tuple) else output.detach()
        return hk
    for i, blk in enumerate(model.transformer.h):
        handles.append(blk.register_forward_hook(make_block_hook(i)))

    # Run in chunks of 1024 tokens
    chunk = 1024
    acts_pre = [[] for _ in range(n_layers)]   # acts_pre[i] = list of (chunk, d_model) tensors
    acts_post = [[] for _ in range(n_layers)]
    for s in range(0, ids.numel(), chunk):
        e = min(s + chunk, ids.numel())
        x = ids[s:e].unsqueeze(0)
        with torch.no_grad():
            model(x)
        for i in range(n_layers):
            acts_pre[i].append(pre_holder[i][0])     # (seq, d)
            acts_post[i].append(post_holder[i][0])
    for h in handles: h.remove()
    acts_pre = [torch.cat(a, dim=0) for a in acts_pre]   # (n_tokens, d_model)
    acts_post = [torch.cat(a, dim=0) for a in acts_post]
    print(f"    captured: pre[0].shape={tuple(acts_pre[0].shape)}", flush=True)

    # ---- For each target layer, run the 3-way comparison ----
    sae_repo = "jbloom/GPT2-Small-SAEs-Reformatted"
    sae_cache = {}
    def get_sae_D(layer_for_sae: int):
        if layer_for_sae in sae_cache: return sae_cache[layer_for_sae]
        folder = f"blocks.{layer_for_sae}.hook_resid_pre"
        path = hf_hub_download(repo_id=sae_repo, filename=f"{folder}/sae_weights.safetensors")
        st = load_file(path)
        W_dec = st["W_dec"]
        if W_dec.shape[1] != d_model: W_dec = W_dec.T
        D = W_dec.to(device=device, dtype=torch.float32).T.contiguous()
        sae_cache[layer_for_sae] = D
        return D

    target_layers = [1, 3, 5, 8, 10]
    g_rand = torch.Generator(device=device).manual_seed(0)
    K_pca = 24576  # match SAE width for fairness in MP step

    results = []
    for L in target_layers:
        print(f"\n[2.L={L}] mlp.c_proj of layer {L}", flush=True)
        # W = mlp.c_proj.weight  shape (4*d_model, d_model) in Conv1D format (in, out).
        # ROWS are output directions in residual space. Rewrite Y = W.T so columns of Y are vectors in R^d_model.
        W = model.transformer.h[L].mlp.c_proj.weight.detach().clone().to(torch.float32)
        Y = W.T.contiguous()                                   # (d_model, 4*d_model)
        n_in = Y.shape[1]

        # ---- SAE dictionary (layer L+1) ----
        L_sae = min(L + 1, n_layers - 1)
        D_sae = get_sae_D(L_sae)
        K_sae = D_sae.shape[1]

        # ---- Activation PCA dictionary from acts_post[L] ----
        # Center, then SVD
        A = acts_post[L].to(torch.float32)                     # (n_tokens, d_model)
        A_centered = A - A.mean(dim=0, keepdim=True)
        # Use truncated SVD via torch.linalg.svd of A (could be slow for huge N; we have ~16k tokens × 768)
        # Top-K_pca right singular vectors of A_centered (rows = tokens, cols = features)
        U_a, S_a, Vh_a = torch.linalg.svd(A_centered, full_matrices=False)
        # principal directions (in feature space) = rows of Vh_a; shape (min(N,d), d)
        # Build PCA dictionary in same orientation as D_sae: (d_model, K_pca)
        # Note: we can have at most min(N, d_model)=d_model PCA directions in d_model space.
        # K_pca > d_model is impossible. To match SAE width we'll repeat-and-perturb? No --
        # better: report PCA at multiple K values up to d_model.
        K_pca_eff = min(K_pca, Vh_a.shape[0])
        D_pca = Vh_a[:K_pca_eff].T.contiguous()                # (d_model, K_pca_eff)

        # ---- Random dict ----
        D_rand = torch.randn(d_model, K_sae, device=device, generator=g_rand, dtype=torch.float32)
        D_rand = D_rand / D_rand.norm(dim=0, keepdim=True).clamp(min=1e-8)

        # ---- Sparse coding for each ----
        def relerr_with(D_dict):
            Z = batched_mp(D_dict, Y, k_sparse)
            Y_hat = D_dict @ Z
            return ((Y - Y_hat).norm() / Y.norm()).item()

        relerr_sae = relerr_with(D_sae)
        relerr_pca = relerr_with(D_pca)
        relerr_rand = relerr_with(D_rand)

        # ---- Rank-k SVD lower bound on linear approximation ----
        U_y, S_y, Vh_y = torch.linalg.svd(Y, full_matrices=False)
        Y_svd = (U_y[:, :k_sparse] * S_y[:k_sparse]) @ Vh_y[:k_sparse, :]
        relerr_svdK = ((Y - Y_svd).norm() / Y.norm()).item()

        # ---- "Activation-aware rank-k": project columns of Y onto top-k principal directions of A ----
        # The columns of Y are output directions in residual stream space. We keep only the
        # components along which activations actually land:
        #   y_j ~= P_pca_topk @ (P_pca_topk.T @ y_j)
        P_pca_topk = Vh_a[:k_sparse].T   # (d_model, k_sparse)
        Y_actk = P_pca_topk @ (P_pca_topk.T @ Y)
        relerr_actk = ((Y - Y_actk).norm() / Y.norm()).item()

        rec = {
            "L": L, "weight": "mlp.c_proj", "L_sae": L_sae,
            "K_sae": K_sae, "K_pca": K_pca_eff, "k_sparse": k_sparse,
            "relerr_sae": relerr_sae,
            "relerr_pca": relerr_pca,
            "relerr_rand": relerr_rand,
            "relerr_svdK": relerr_svdK,
            "relerr_actk": relerr_actk,
        }
        results.append(rec)
        print(f"   L={L} mlp.c_proj sae=L{L_sae}  "
              f"sae={relerr_sae:.4f}  pca={relerr_pca:.4f}  rand={relerr_rand:.4f}  "
              f"svdK={relerr_svdK:.4f}  actK={relerr_actk:.4f}", flush=True)

    print("\n=== SUMMARY (lower is better) ===", flush=True)
    print(f"{'L':>3} {'sae':>8} {'pca':>8} {'rand':>8} {'svdK':>8} {'actK':>8}", flush=True)
    for r in results:
        print(f"{r['L']:>3}  {r['relerr_sae']:.4f}  {r['relerr_pca']:.4f}  "
              f"{r['relerr_rand']:.4f}  {r['relerr_svdK']:.4f}  {r['relerr_actk']:.4f}", flush=True)

    # Verdict
    sae_wins_pca = sum(1 for r in results if r['relerr_sae'] < r['relerr_pca'] - 0.01)
    pca_wins_sae = sum(1 for r in results if r['relerr_pca'] < r['relerr_sae'] - 0.01)
    print(f"\nSAE beats PCA by >1pp: {sae_wins_pca}/{len(results)}", flush=True)
    print(f"PCA beats SAE by >1pp: {pca_wins_sae}/{len(results)}", flush=True)
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
def main(k_sparse: int = 32, n_activation_tokens: int = 16384):
    res = run.remote(k_sparse=k_sparse, n_activation_tokens=n_activation_tokens)
    print(res)
