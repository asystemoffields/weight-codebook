"""
Weight-codebook probe (GPT-2 small variant).

Pivoted from Gemma 2 9B (gated). The structural hypothesis is the same: do the
columns of W_O live in span(D) where D is the SAE decoder for that layer's
residual stream? If yes, weight matrices factor over a learned universal
feature basis -> compression beyond what numerical-statistics PTQ achieves.

Model: gpt2 (124M, d_model=768, 12 layers, 12 heads, head_dim=64)
SAE:   jbloom/GPT2-Small-SAEs-Reformatted, blocks.{L}.hook_resid_pre (K=24576)
"""
import modal

app = modal.App("weight-codebook-probe")

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
def probe(
    layer_idx: int = 5,
    k_sparse: int = 32,
    n_eval_tokens: int = 32768,
):
    import os, math, time, json
    import torch, numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    device = "cuda"
    torch.set_grad_enabled(False)

    print(f"=== Weight-codebook probe (GPT-2 small) ===")
    print(f"  layer={layer_idx}  k_sparse={k_sparse}  n_eval_tokens={n_eval_tokens}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # ----- 1. Load GPT-2 small -----
    print("\n[1] loading gpt2...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", torch_dtype=torch.float32, device_map="cuda",
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained("gpt2")
    print(f"    loaded in {time.time()-t0:.1f}s", flush=True)
    cfg = model.config
    d_model = cfg.n_embd
    print(f"    d_model={d_model}  n_layers={cfg.n_layer}  n_heads={cfg.n_head}", flush=True)

    # ----- 2. Load SAE -----
    sae_repo = "jbloom/GPT2-Small-SAEs-Reformatted"
    sae_folder = f"blocks.{layer_idx}.hook_resid_pre"
    print(f"\n[2] loading SAE from {sae_repo}/{sae_folder}...", flush=True)
    t0 = time.time()
    cfg_path = hf_hub_download(repo_id=sae_repo, filename=f"{sae_folder}/cfg.json")
    with open(cfg_path) as f: sae_cfg = json.load(f)
    print(f"    cfg keys: {list(sae_cfg.keys())[:8]}...", flush=True)

    weights_path = hf_hub_download(repo_id=sae_repo, filename=f"{sae_folder}/sae_weights.safetensors")
    state = load_file(weights_path)
    print(f"    state keys: {list(state.keys())}", flush=True)
    # Typical sae-lens keys: W_enc, W_dec, b_enc, b_dec (and maybe b_dec_out)
    W_dec = state["W_dec"]  # expected (K, d_model)
    if W_dec.shape[0] == d_model and W_dec.shape[1] != d_model:
        W_dec = W_dec.T  # be defensive about orientation
    K = W_dec.shape[0]
    print(f"    W_dec shape: ({K}, {d_model})  load time: {time.time()-t0:.1f}s", flush=True)
    D = W_dec.to(device=device, dtype=torch.float32).T.contiguous()  # (d_model, K)

    # ----- 3. Extract W_O and reconstruct -----
    print(f"\n[3] extracting GPT-2 layer {layer_idx} attn.c_proj weight (W_O equivalent)...", flush=True)
    # In HF GPT2, the output projection of an attention block is `attn.c_proj`,
    # implemented as Conv1D with weight shape (d_model, d_model) where the linear
    # operation is y = x @ W (not W @ x). So columns of c_proj.weight are the
    # output directions produced by each input dim.
    c_proj = model.transformer.h[layer_idx].attn.c_proj
    W_O = c_proj.weight.detach().clone().to(torch.float32)  # (d_model, d_model)
    # Treat each column as a vector in residual-stream output space.
    print(f"    W_O shape: {tuple(W_O.shape)}  ||W_O||_F = {W_O.norm().item():.4f}", flush=True)

    t0 = time.time()
    Z = batched_matching_pursuit(D, W_O, k_sparse)
    W_O_recon = (D @ Z).to(torch.float32)
    relerr_cb = ((W_O - W_O_recon).norm() / W_O.norm()).item()
    print(f"    MP done in {time.time()-t0:.2f}s", flush=True)
    print(f"    relative Frobenius error (codebook): {relerr_cb:.6f}", flush=True)

    # ----- 4. SVD baseline at matched bytes -----
    d_out, d_in = W_O.shape
    codebook_bytes = 4 * k_sparse * d_in  # 2-byte index + 2-byte value per nz
    rank_for_match = max(1, codebook_bytes // (2 * (d_out + d_in)))
    print(f"\n[4] matched-budget SVD baseline (rank {rank_for_match})...", flush=True)
    U, S_sv, Vh = torch.linalg.svd(W_O, full_matrices=False)
    W_O_svd = (U[:, :rank_for_match] * S_sv[:rank_for_match]) @ Vh[:rank_for_match, :]
    relerr_svd = ((W_O - W_O_svd).norm() / W_O.norm()).item()
    print(f"    SVD relative error: {relerr_svd:.6f}", flush=True)

    # ----- 5. Random gaussian dictionary sanity -----
    print(f"\n[5] random gaussian dict at same K ({K}) — sanity baseline...", flush=True)
    g = torch.Generator(device=device).manual_seed(0)
    D_rand = torch.randn(d_model, K, device=device, generator=g, dtype=torch.float32)
    D_rand = D_rand / D_rand.norm(dim=0, keepdim=True).clamp(min=1e-8)
    Z_rand = batched_matching_pursuit(D_rand, W_O, k_sparse)
    W_O_rand = (D_rand @ Z_rand).to(torch.float32)
    relerr_rand = ((W_O - W_O_rand).norm() / W_O.norm()).item()
    print(f"    random-dict relative error: {relerr_rand:.6f}", flush=True)

    # ----- 6. PPL eval -----
    print(f"\n[6] PPL eval (n_tokens={n_eval_tokens})...", flush=True)
    W_orig = c_proj.weight.detach().clone()
    ppl_orig = eval_ppl(model, tok, n_eval_tokens)
    print(f"    baseline PPL: {ppl_orig:.4f}", flush=True)

    c_proj.weight.data.copy_(W_O_recon.T if c_proj.weight.shape != W_O_recon.shape else W_O_recon)
    # GPT2 Conv1D weight is (in, out) = (d_model, d_model); since both are equal,
    # the transpose ambiguity is invisible. Just copy.
    c_proj.weight.data.copy_(W_O_recon)
    ppl_cb = eval_ppl(model, tok, n_eval_tokens)
    print(f"    codebook PPL: {ppl_cb:.4f}  (delta {ppl_cb - ppl_orig:+.4f})", flush=True)

    c_proj.weight.data.copy_(W_O_svd)
    ppl_svd = eval_ppl(model, tok, n_eval_tokens)
    print(f"    SVD PPL: {ppl_svd:.4f}  (delta {ppl_svd - ppl_orig:+.4f})", flush=True)

    c_proj.weight.data.copy_(W_O_rand)
    ppl_rand = eval_ppl(model, tok, n_eval_tokens)
    print(f"    random-dict PPL: {ppl_rand:.4f}  (delta {ppl_rand - ppl_orig:+.4f})", flush=True)

    c_proj.weight.data.copy_(W_orig)

    out = {
        "model": "gpt2",
        "sae_repo": sae_repo,
        "sae_folder": sae_folder,
        "layer_idx": layer_idx,
        "k_sparse": k_sparse,
        "K": K, "d_model": d_model,
        "n_eval_tokens": n_eval_tokens,
        "rank_svd_matched": rank_for_match,
        "relerr_codebook": relerr_cb,
        "relerr_svd": relerr_svd,
        "relerr_random": relerr_rand,
        "ppl_orig": ppl_orig,
        "ppl_codebook": ppl_cb,
        "ppl_svd": ppl_svd,
        "ppl_random": ppl_rand,
        "delta_codebook": ppl_cb - ppl_orig,
        "delta_svd": ppl_svd - ppl_orig,
        "delta_random": ppl_rand - ppl_orig,
    }
    print("\n=== RESULT ===", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    return out


def batched_matching_pursuit(D, Y, k):
    """Greedy MP: Y[:,j] ~= D @ Z[:,j] with up to k nonzeros."""
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


def eval_ppl(model, tok, n_tokens, seq_len=1024):
    """Standard WikiText-2 perplexity (non-overlapping windows)."""
    import math, torch
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tok(text, return_tensors="pt")
    input_ids = enc.input_ids[0].to(model.device)
    if input_ids.numel() < n_tokens:
        n_tokens = input_ids.numel() - seq_len
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
def main(layer_idx: int = 5, k_sparse: int = 32, n_eval_tokens: int = 32768):
    result = probe.remote(layer_idx=layer_idx, k_sparse=k_sparse, n_eval_tokens=n_eval_tokens)
    print(result)
