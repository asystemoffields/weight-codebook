"""
End-to-end compression probe.

Earlier probes confirmed: SAE atoms span W column space substantially better than
random/PCA/SVD. The natural next question: **does this translate to actual bytes
saved at usable quality?**

This probe:
  1. For each MLP down-projection W in Pythia 410M (24 layers), sparse-code W
     over the matching SAE decoder atoms with k_sparse atoms per column.
  2. Quantize the sparse codes: indices to ceil(log2(K)) bits, values to 8-bit.
  3. Substitute reconstructed W back into the model.
  4. Measure WikiText-2 PPL of the patched model.
  5. Sum bytes: shared dict (amortized) + per-matrix sparse codes.

Compare to:
  - FP32 / FP16 baselines (orig PPL)
  - INT8 / INT4 simple per-matrix quant (matched-byte references)

The goal: find the smallest bytes-per-MLP-matrix at which PPL stays within
some threshold (e.g., 1.5x baseline) and confirm whether SAE-codebook beats
direct quantization at the same byte budget.
"""
import modal

app = modal.App("probe-compression")

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
def run(k_sparse: int = 32, value_bits: int = 8, n_eval_tokens: int = 8192,
        layers_to_compress: str = "all_mlp_down"):
    """layers_to_compress: comma-separated layer indices, or 'all_mlp_down' for all."""
    import os, math, json, time
    import torch, numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from datasets import load_dataset

    device = "cuda"
    torch.set_grad_enabled(False)

    print(f"=== End-to-end compression probe ===", flush=True)
    print(f"  k_sparse={k_sparse}  value_bits={value_bits}  n_eval_tokens={n_eval_tokens}", flush=True)
    print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # ----- Load model -----
    print("\n[1] loading pythia-410m...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-410m", torch_dtype=torch.float32, device_map="cuda",
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    cfg = model.config
    d_model = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
    intermediate = cfg.intermediate_size
    print(f"    loaded in {time.time()-t0:.1f}s; d_model={d_model} n_layers={n_layers} intermediate={intermediate}", flush=True)

    # ----- Pick layers to compress -----
    if layers_to_compress == "all_mlp_down":
        target_layers = list(range(n_layers))
    else:
        target_layers = [int(x) for x in layers_to_compress.split(",")]
    print(f"    compressing layers: {target_layers}", flush=True)

    # ----- Baseline PPL -----
    print("\n[2] baseline PPL...", flush=True)
    ppl_orig = eval_ppl(model, tok, n_eval_tokens)
    print(f"    baseline PPL: {ppl_orig:.4f}", flush=True)

    # ----- Reconstruct W for each layer -----
    sae_repo = "EleutherAI/sae-pythia-410m-65k"
    backups = {}
    total_codebook_bytes = 0
    total_dict_bytes_unique = 0   # bytes if we count each unique dictionary once
    layer_relerrs = []

    for L in target_layers:
        print(f"\n[3.{L}] compressing layer {L} dense_4h_to_h", flush=True)
        t0 = time.time()
        path = hf_hub_download(repo_id=sae_repo, filename=f"layers.{L}.mlp/sae.safetensors")
        st = load_file(path)
        W_dec = st["W_dec"]
        if W_dec.shape[1] != d_model: W_dec = W_dec.T
        D = W_dec.to(device=device, dtype=torch.float32).T.contiguous()
        K = D.shape[1]
        total_dict_bytes_unique += K * d_model * 4   # FP32 dict per unique
        print(f"    SAE loaded ({time.time()-t0:.1f}s)  D=({d_model}, {K})", flush=True)

        block = model.gpt_neox.layers[L]
        W = block.mlp.dense_4h_to_h.weight.detach().clone().to(torch.float32)
        backups[L] = W.clone()

        # Sparse code
        Z = batched_mp(D, W, k_sparse)
        # Quantize values to value_bits per nz; indices need ceil(log2(K))=16 bits
        nz_mask = Z != 0
        # per-column scale
        col_max = Z.abs().amax(dim=0, keepdim=True).clamp(min=1e-12)
        q_levels = (1 << (value_bits - 1)) - 1
        Z_q = torch.round(Z / col_max * q_levels).clamp(-q_levels, q_levels)
        Z_deq = (Z_q / q_levels) * col_max
        W_recon = D @ Z_deq

        relerr_full = ((W - (D @ Z)).norm() / W.norm()).item()
        relerr_qnt = ((W - W_recon).norm() / W.norm()).item()
        print(f"    relerr (continuous): {relerr_full:.4f}", flush=True)
        print(f"    relerr (quantized {value_bits}-bit values): {relerr_qnt:.4f}", flush=True)

        # Substitute
        block.mlp.dense_4h_to_h.weight.data.copy_(W_recon)
        layer_relerrs.append((L, relerr_full, relerr_qnt))

        n_in = W.shape[1]
        index_bits = max(1, math.ceil(math.log2(K)))
        bytes_per_matrix = (k_sparse * n_in * (index_bits + value_bits) + 7) // 8 + 8  # +scales
        total_codebook_bytes += bytes_per_matrix

    # ----- Evaluate the patched model -----
    print(f"\n[4] PPL after substituting all {len(target_layers)} mlp.dense_4h_to_h with codebook reconstructions...", flush=True)
    ppl_codebook = eval_ppl(model, tok, n_eval_tokens)
    print(f"    codebook PPL: {ppl_codebook:.4f}  (delta {ppl_codebook - ppl_orig:+.4f})", flush=True)

    # Restore for INT8 baseline
    for L in target_layers:
        model.gpt_neox.layers[L].mlp.dense_4h_to_h.weight.data.copy_(backups[L])

    # ----- INT8 / INT4 per-matrix quant baselines for comparison -----
    print(f"\n[5] simple INT8 per-row quant baseline...", flush=True)
    for L in target_layers:
        W = backups[L]
        scale = W.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 127
        Wq = torch.round(W / scale).clamp(-127, 127) * scale
        model.gpt_neox.layers[L].mlp.dense_4h_to_h.weight.data.copy_(Wq)
    ppl_int8 = eval_ppl(model, tok, n_eval_tokens)
    print(f"    INT8 PPL: {ppl_int8:.4f}  (delta {ppl_int8 - ppl_orig:+.4f})", flush=True)

    print(f"\n[6] simple INT4 per-row quant baseline...", flush=True)
    for L in target_layers:
        W = backups[L]
        scale = W.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 7
        Wq = torch.round(W / scale).clamp(-7, 7) * scale
        model.gpt_neox.layers[L].mlp.dense_4h_to_h.weight.data.copy_(Wq)
    ppl_int4 = eval_ppl(model, tok, n_eval_tokens)
    print(f"    INT4 PPL: {ppl_int4:.4f}  (delta {ppl_int4 - ppl_orig:+.4f})", flush=True)

    for L in target_layers:
        model.gpt_neox.layers[L].mlp.dense_4h_to_h.weight.data.copy_(backups[L])

    # ----- Byte accounting -----
    one_W_bytes_fp32 = d_model * intermediate * 4
    one_W_bytes_int8 = d_model * intermediate * 1
    one_W_bytes_int4 = (d_model * intermediate + 1) // 2
    per_matrix_codebook_bytes = total_codebook_bytes // len(target_layers)
    avg_dict_bytes_per_layer_amortized = total_dict_bytes_unique // len(target_layers)

    print(f"\n=== BYTE ACCOUNTING (per W matrix, dense_4h_to_h) ===", flush=True)
    print(f"  FP32 baseline:           {one_W_bytes_fp32:,} bytes", flush=True)
    print(f"  INT8 baseline:           {one_W_bytes_int8:,} bytes", flush=True)
    print(f"  INT4 baseline:           {one_W_bytes_int4:,} bytes", flush=True)
    print(f"  Codebook (per-layer SAE):  {per_matrix_codebook_bytes:,} bytes + {avg_dict_bytes_per_layer_amortized:,} bytes dict (NOT amortized)", flush=True)
    print(f"  Codebook + amortized dict (if one dict serves all 24 layers): "
          f"{per_matrix_codebook_bytes + total_dict_bytes_unique // len(target_layers) // 24:,} bytes  "
          f"(speculative — requires cross-layer test)", flush=True)

    out = {
        "n_layers_compressed": len(target_layers),
        "k_sparse": k_sparse, "value_bits": value_bits,
        "ppl_orig": ppl_orig,
        "ppl_codebook": ppl_codebook,
        "ppl_int8": ppl_int8,
        "ppl_int4": ppl_int4,
        "delta_codebook": ppl_codebook - ppl_orig,
        "delta_int8": ppl_int8 - ppl_orig,
        "delta_int4": ppl_int4 - ppl_orig,
        "per_matrix_codebook_bytes": per_matrix_codebook_bytes,
        "per_matrix_int8_bytes": one_W_bytes_int8,
        "per_matrix_int4_bytes": one_W_bytes_int4,
        "per_matrix_fp32_bytes": one_W_bytes_fp32,
        "dict_bytes_total_fp32": total_dict_bytes_unique,
        "layer_relerrs": [(L, ce, qe) for (L, ce, qe) in layer_relerrs],
    }
    print(f"\n=== RESULT ===", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    return out


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


def eval_ppl(model, tok, n_tokens, seq_len=1024):
    import math, torch
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tok(text, return_tensors="pt")
    input_ids = enc.input_ids[0].to(model.device)
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
def main(k_sparse: int = 32, value_bits: int = 8, n_eval_tokens: int = 8192,
         layers: str = "all_mlp_down"):
    res = run.remote(k_sparse=k_sparse, value_bits=value_bits,
                      n_eval_tokens=n_eval_tokens, layers_to_compress=layers)
    print(res)
