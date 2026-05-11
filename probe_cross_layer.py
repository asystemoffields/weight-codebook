"""
Cross-layer codebook probe.

If `probe.py` (same-layer) lands a KEEP, the next question for real compression
is: does a SINGLE dictionary D, taken from one layer's SAE, also sparse-code
the weight matrices of OTHER layers?

If yes -> one dictionary spans many layers -> the per-layer overhead drops
from ~117 MB to amortized ~3 MB across 42 layers in Gemma 2 9B. That's the
difference between "interesting fingerprint" and "category jump."

This script: pick a source layer L_src, use its SAE residual-stream decoder as
the universal D, then for each target layer L_tgt in {L_src-5, L_src, L_src+5,
L_src+10}, sparse-code W_O of L_tgt over D and measure:

  - reconstruction Frobenius relative error
  - PPL delta after substitution

Compared against:
  - same-layer baseline (L_tgt == L_src) — should be best
  - matched-budget SVD per target layer
"""
import modal

app = modal.App("weight-codebook-cross-layer")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2.0",
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate>=1.0.0",
        "datasets>=3.0.0,<4.0.0",
        "huggingface_hub[hf_transfer]>=0.26,<1.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

hf_cache = modal.Volume.from_name("hf-cache-weight-codebook", create_if_missing=True)


@app.function(
    image=image, gpu="H100", timeout=3600,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface")],
)
def probe_cross(
    layer_src: int = 20,
    layer_targets: list = None,
    k_sparse: int = 32,
    n_eval_tokens: int = 8192,
    sae_width: str = "width_16k",
):
    import os, math, time, json
    import torch, numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import HfApi, hf_hub_download

    device = "cuda"
    torch.set_grad_enabled(False)

    if layer_targets is None:
        layer_targets = [layer_src - 5, layer_src, layer_src + 5, layer_src + 10]
    layer_targets = [int(L) for L in layer_targets if 0 <= int(L) < 42]

    print(f"=== Cross-layer codebook probe ===")
    print(f"  source SAE layer: {layer_src}")
    print(f"  target W_O layers: {layer_targets}")
    print(f"  k_sparse={k_sparse}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ---- 1. Load Gemma 2 9B ----
    print("\n[1] loading Gemma 2 9B...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b", torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="eager",
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained("google/gemma-2-9b")
    print(f"    loaded in {time.time()-t0:.1f}s")

    # ---- 2. Load source-layer SAE ----
    print(f"\n[2] loading SAE from layer {layer_src}...")
    t0 = time.time()
    repo_id = "google/gemma-scope-9b-pt-res"
    api = HfApi()
    files = api.list_repo_files(repo_id)
    prefix = f"layer_{layer_src}/{sae_width}/"
    matching = [f for f in files if f.startswith(prefix) and f.endswith("params.npz")]
    def l0_of(f):
        try: return int(f.split("average_l0_")[1].split("/")[0])
        except: return 10**9
    matching.sort(key=l0_of)
    chosen = matching[len(matching) // 2]
    print(f"    chosen: {chosen}")
    local_npz = hf_hub_download(repo_id=repo_id, filename=chosen)
    W_dec_np = np.load(local_npz)["W_dec"]   # (K, d_model)
    K, d_model = W_dec_np.shape
    D = torch.from_numpy(W_dec_np).T.contiguous().to(device=device, dtype=torch.float32)
    print(f"    D shape: ({d_model}, {K})  load time: {time.time()-t0:.1f}s")

    # ---- 3. Baseline PPL ----
    print(f"\n[3] baseline PPL...")
    ppl_orig = eval_ppl(model, tok, n_eval_tokens)
    print(f"    baseline PPL: {ppl_orig:.4f}")

    # ---- 4. Sweep target layers ----
    rows = []
    backup = {}
    for L_tgt in layer_targets:
        print(f"\n[4.{L_tgt}] reconstructing W_O of layer {L_tgt} from layer-{layer_src} dict...")
        W_orig = model.model.layers[L_tgt].self_attn.o_proj.weight.detach().clone()
        backup[L_tgt] = W_orig
        W = W_orig.to(torch.float32)
        d_out, d_in = W.shape

        # codebook reconstruction
        Z = batched_matching_pursuit(D, W, k_sparse)
        W_rec = (D @ Z).to(torch.float32)
        rel_err = ((W - W_rec).norm() / W.norm()).item()

        # matched-budget SVD
        codebook_bytes = 4 * k_sparse * d_in
        rank_for_match = max(1, codebook_bytes // (2 * (d_out + d_in)))
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        W_svd = (U[:, :rank_for_match] * S[:rank_for_match]) @ Vh[:rank_for_match, :]
        svd_err = ((W - W_svd).norm() / W.norm()).item()

        # substitute codebook reconstruction; eval
        model.model.layers[L_tgt].self_attn.o_proj.weight.data.copy_(W_rec.to(torch.bfloat16))
        ppl_cb = eval_ppl(model, tok, n_eval_tokens)
        # substitute SVD; eval
        model.model.layers[L_tgt].self_attn.o_proj.weight.data.copy_(W_svd.to(torch.bfloat16))
        ppl_svd = eval_ppl(model, tok, n_eval_tokens)
        # restore
        model.model.layers[L_tgt].self_attn.o_proj.weight.data.copy_(W_orig)

        rec = {
            "layer_src": layer_src,
            "layer_tgt": L_tgt,
            "k_sparse": k_sparse,
            "rank_svd": rank_for_match,
            "relerr_codebook": rel_err,
            "relerr_svd": svd_err,
            "ppl_orig": ppl_orig,
            "ppl_codebook": ppl_cb,
            "ppl_svd": ppl_svd,
            "delta_codebook": ppl_cb - ppl_orig,
            "delta_svd": ppl_svd - ppl_orig,
        }
        print(f"    layer_tgt={L_tgt}  relerr_cb={rel_err:.4f}  relerr_svd={svd_err:.4f}  "
              f"ppl_cb={ppl_cb:.4f}({rec['delta_codebook']:+.4f})  ppl_svd={ppl_svd:.4f}({rec['delta_svd']:+.4f})")
        rows.append(rec)

    print("\n=== CROSS-LAYER RESULT ===")
    print(json.dumps(rows, indent=2))
    return rows


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


def eval_ppl(model, tok, n_tokens, seq_len=2048):
    import math, torch
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tok(text, return_tensors="pt")
    input_ids = enc.input_ids[0, : max(seq_len, n_tokens + seq_len)].to(model.device)
    total_nll = 0.0; total_count = 0; pos = 0
    while pos + seq_len <= input_ids.numel() and total_count < n_tokens:
        chunk = input_ids[pos : pos + seq_len].unsqueeze(0)
        with torch.no_grad():
            out = model(chunk, labels=chunk)
        total_nll += out.loss.item() * (seq_len - 1)
        total_count += (seq_len - 1)
        pos += seq_len
    return math.exp(total_nll / max(1, total_count))


@app.local_entrypoint()
def main(layer_src: int = 20, k_sparse: int = 32):
    result = probe_cross.remote(layer_src=layer_src, k_sparse=k_sparse)
    print(result)
