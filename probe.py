"""
Weight-codebook probe: can a single Gemma Scope SAE dictionary sparse-code the
W_O weight matrix of the same layer, at byte-budgets competitive with low-rank
SVD, while preserving model perplexity?

Hypothesis: columns of W_O live in span(D) where D is the SAE decoder for the
residual stream at that layer. If yes, W ~= D @ Z with Z sparse, and we get
~5-10x compression beyond standard 4-bit quant by exploiting the *computational*
basis rather than the *numerical* one.

Cheap first test: one layer, one matrix (o_proj), Gemma 2 9B.
"""
import modal

app = modal.App("weight-codebook-probe")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.46.0",
        "accelerate>=1.0.0",
        "datasets>=3.0.0",
        "sae-lens>=4.0.0",
        "numpy",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

hf_cache = modal.Volume.from_name("hf-cache-weight-codebook", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface")],
)
def probe(
    layer_idx: int = 20,
    k_sparse: int = 32,
    n_eval_tokens: int = 16384,
    sae_width: str = "width_16k",
):
    import os, math, time, json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    device = "cuda"
    torch.set_grad_enabled(False)

    print(f"=== Weight-codebook probe ===")
    print(f"  layer={layer_idx}  k_sparse={k_sparse}  n_eval_tokens={n_eval_tokens}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.mem_get_info()[1]/1e9:.1f} GB")

    # ----- 1. Load model -----
    print("\n[1] loading Gemma 2 9B...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained("google/gemma-2-9b")
    print(f"    loaded in {time.time()-t0:.1f}s")
    print(f"    d_model={model.config.hidden_size}  n_layers={model.config.num_hidden_layers}")

    # ----- 2. Load SAE for residual stream at layer_idx -----
    print(f"\n[2] loading Gemma Scope SAE (residual stream, layer {layer_idx})...")
    t0 = time.time()
    from sae_lens import SAE
    from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

    release = "gemma-scope-9b-pt-res"
    directory = get_pretrained_saes_directory()
    available = directory[release].saes_map  # {sae_id: huggingface_path}
    # pick the layer's SAEs and the lowest L0 (sparsest) available for the chosen width
    candidates = [k for k in available.keys() if k.startswith(f"layer_{layer_idx}/{sae_width}/")]
    print(f"    {len(candidates)} SAEs available at layer {layer_idx}, {sae_width}")
    # Sort by L0 (smaller is sparser) -- prefer lowest reasonable L0 for cleanest atoms
    def l0_of(sid):
        try:
            return int(sid.rsplit("_", 1)[-1])
        except Exception:
            return 10**9
    candidates.sort(key=l0_of)
    if not candidates:
        raise RuntimeError(f"No SAEs found for layer_{layer_idx}/{sae_width}")
    sae_id = candidates[len(candidates) // 2]  # median L0 -- not too sparse, not too dense
    print(f"    chosen: {sae_id}")

    sae, cfg_dict, _ = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    sae = sae.to(torch.float32)
    # W_dec shape in sae_lens: (d_sae, d_in) -- rows are decoder atoms
    W_dec = sae.W_dec  # (K, d_model)
    K, d_model = W_dec.shape
    print(f"    SAE W_dec shape: ({K}, {d_model})  load time: {time.time()-t0:.1f}s")

    # Dictionary as columns: D in R^{d_model x K}
    D = W_dec.T.contiguous().to(torch.float32)
    # ----- 3. Extract W_O and run sparse coding -----
    print(f"\n[3] extracting W_O of layer {layer_idx} and sparse-coding over D...")
    W_O_orig = model.model.layers[layer_idx].self_attn.o_proj.weight.detach().clone()
    W_O = W_O_orig.to(torch.float32)
    print(f"    W_O shape: {tuple(W_O.shape)}  ||W_O||_F = {W_O.norm().item():.4f}")

    t0 = time.time()
    Z = batched_matching_pursuit(D, W_O, k_sparse)
    W_O_recon = (D @ Z).to(torch.float32)
    recon_err = ((W_O - W_O_recon).norm() / W_O.norm()).item()
    print(f"    MP done in {time.time()-t0:.2f}s")
    print(f"    relative Frobenius error (codebook): {recon_err:.6f}")

    # ----- 4. SVD baseline at matched bytes -----
    print(f"\n[4] computing SVD baseline at matched byte budget...")
    # Bytes per column: k_sparse * (2 bytes index + 2 bytes value) = 4*k_sparse
    # Total for codebook (not counting shared D): 4 * k_sparse * d_in
    # Equivalent rank-r SVD storage: r * (d_out + d_in) * 2 bytes
    d_out, d_in = W_O.shape
    codebook_bytes = 4 * k_sparse * d_in
    rank_for_match = max(1, codebook_bytes // (2 * (d_out + d_in)))
    print(f"    codebook bytes/matrix: {codebook_bytes:,}")
    print(f"    matched SVD rank: {rank_for_match}")

    t0 = time.time()
    U, S_sv, Vh = torch.linalg.svd(W_O, full_matrices=False)
    W_O_svd = (U[:, :rank_for_match] * S_sv[:rank_for_match]) @ Vh[:rank_for_match, :]
    svd_err = ((W_O - W_O_svd).norm() / W_O.norm()).item()
    print(f"    SVD rank-{rank_for_match} done in {time.time()-t0:.2f}s")
    print(f"    relative Frobenius error (SVD): {svd_err:.6f}")

    # ----- 5. PPL eval -----
    print(f"\n[5] PPL eval (n_tokens={n_eval_tokens})")
    print(f"    [a] baseline (untouched model)...")
    ppl_orig = eval_ppl(model, tok, n_eval_tokens)
    print(f"        baseline PPL: {ppl_orig:.4f}")

    print(f"    [b] codebook-reconstructed W_O (k_sparse={k_sparse})...")
    model.model.layers[layer_idx].self_attn.o_proj.weight.data.copy_(W_O_recon.to(torch.bfloat16))
    ppl_codebook = eval_ppl(model, tok, n_eval_tokens)
    print(f"        codebook PPL: {ppl_codebook:.4f}  (delta {ppl_codebook - ppl_orig:+.4f})")

    print(f"    [c] SVD rank-{rank_for_match} W_O at matched bytes...")
    model.model.layers[layer_idx].self_attn.o_proj.weight.data.copy_(W_O_svd.to(torch.bfloat16))
    ppl_svd = eval_ppl(model, tok, n_eval_tokens)
    print(f"        SVD PPL: {ppl_svd:.4f}  (delta {ppl_svd - ppl_orig:+.4f})")

    # ----- 6. Sanity: random gaussian dictionary at same K -----
    print(f"    [d] sanity: random gaussian dictionary at same K (should be much worse)...")
    g = torch.Generator(device=device).manual_seed(0)
    D_rand = torch.randn(d_model, K, device=device, generator=g, dtype=torch.float32)
    D_rand = D_rand / D_rand.norm(dim=0, keepdim=True).clamp(min=1e-8)
    Z_rand = batched_matching_pursuit(D_rand, W_O, k_sparse)
    W_O_rand = (D_rand @ Z_rand).to(torch.float32)
    rand_err = ((W_O - W_O_rand).norm() / W_O.norm()).item()
    model.model.layers[layer_idx].self_attn.o_proj.weight.data.copy_(W_O_rand.to(torch.bfloat16))
    ppl_rand = eval_ppl(model, tok, n_eval_tokens)
    print(f"        random-dict relerr: {rand_err:.6f}  PPL: {ppl_rand:.4f}  (delta {ppl_rand - ppl_orig:+.4f})")

    # restore
    model.model.layers[layer_idx].self_attn.o_proj.weight.data.copy_(W_O_orig)

    out = {
        "release": release,
        "sae_id": sae_id,
        "layer_idx": layer_idx,
        "k_sparse": k_sparse,
        "K": K,
        "d_model": d_model,
        "n_eval_tokens": n_eval_tokens,
        "rank_for_match": rank_for_match,
        "recon_relerr_codebook": recon_err,
        "recon_relerr_svd": svd_err,
        "recon_relerr_random": rand_err,
        "ppl_orig": ppl_orig,
        "ppl_codebook": ppl_codebook,
        "ppl_svd": ppl_svd,
        "ppl_random": ppl_rand,
    }
    print("\n=== RESULT ===")
    print(json.dumps(out, indent=2))
    return out


def batched_matching_pursuit(D: "torch.Tensor", Y: "torch.Tensor", k: int) -> "torch.Tensor":
    """
    Greedy matching pursuit (no re-LS): for each column Y[:, j], find a sparse
    code Z[:, j] with at most k nonzeros such that Y[:, j] ~= D @ Z[:, j].

    D: (d, K)
    Y: (d, n)
    returns Z: (K, n) with up to k nonzeros per column.
    """
    import torch
    d, K = D.shape
    _, n = Y.shape
    device = Y.device
    dtype = torch.float32

    D = D.to(dtype)
    Y = Y.to(dtype)
    D_norms = D.norm(dim=0).clamp(min=1e-8)              # (K,)
    D_unit = D / D_norms.unsqueeze(0)                    # (d, K)

    residual = Y.clone()
    Z = torch.zeros(K, n, device=device, dtype=dtype)
    col_idx = torch.arange(n, device=device)

    for step in range(k):
        corr = D_unit.T @ residual                       # (K, n)
        atom = corr.abs().argmax(dim=0)                  # (n,)
        coeff_unit = corr[atom, col_idx]                 # (n,)
        residual = residual - D_unit[:, atom] * coeff_unit.unsqueeze(0)
        Z[atom, col_idx] = Z[atom, col_idx] + coeff_unit / D_norms[atom]

    return Z


def eval_ppl(model, tok, n_tokens: int, seq_len: int = 2048) -> float:
    import math, torch
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tok(text, return_tensors="pt")
    input_ids = enc.input_ids[0, : max(seq_len, n_tokens + seq_len)].to(model.device)
    if input_ids.numel() < n_tokens + seq_len:
        # truncate target
        n_tokens = input_ids.numel() - seq_len
    total_nll = 0.0
    total_count = 0
    pos = 0
    while pos + seq_len <= input_ids.numel() and total_count < n_tokens:
        chunk = input_ids[pos : pos + seq_len].unsqueeze(0)
        with torch.no_grad():
            out = model(chunk, labels=chunk)
        total_nll += out.loss.item() * (seq_len - 1)
        total_count += (seq_len - 1)
        pos += seq_len
    return math.exp(total_nll / max(1, total_count))


@app.local_entrypoint()
def main(layer_idx: int = 20, k_sparse: int = 32, n_eval_tokens: int = 16384):
    result = probe.remote(
        layer_idx=layer_idx,
        k_sparse=k_sparse,
        n_eval_tokens=n_eval_tokens,
    )
    print(result)
