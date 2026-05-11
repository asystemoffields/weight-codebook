"""
Hybrid codebook + INT4 residual probe.

The pure codebook substitution destroyed PPL (19.5 -> 81586). The reason is
per-matrix relerr ~0.55 compounds across 24 layers. But the codebook DOES
capture *some* structure (it beats random by 20pp).

Hybrid idea: store
  W' = D @ Z_q  + INT4_quantize(W - D @ Z_q)

where the codebook handles the "easy" directions and a quantized residual
catches what the codebook missed.

The question: is the residual easier to quantize than W itself? If the
codebook removed the dominant structure, the residual might be closer to
white noise and quantize cheaper.

Compare to plain INT4 at the same total bytes.
"""
import modal

app = modal.App("probe-residual")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2.0", "torch==2.5.1", "transformers==4.46.3",
        "accelerate>=1.0.0", "datasets>=3.0.0,<4.0.0",
        "safetensors>=0.4.5,<1.0",
        "huggingface_hub[hf_transfer]>=0.26,<1.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
)
hf_cache = modal.Volume.from_name("hf-cache-weight-codebook", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_name("huggingface")])
def run(k_sparse: int = 8, residual_bits: int = 4, n_eval_tokens: int = 8192):
    import os, math, json, time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from datasets import load_dataset

    device = "cuda"
    torch.set_grad_enabled(False)
    print(f"=== Hybrid codebook + INT{residual_bits} residual probe ===  k_sparse={k_sparse}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-410m", torch_dtype=torch.float32, device_map="cuda",
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers

    print("[1] baseline PPL...", flush=True)
    ppl_orig = eval_ppl(model, tok, n_eval_tokens)
    print(f"    baseline PPL: {ppl_orig:.4f}", flush=True)

    backups = {L: model.gpt_neox.layers[L].mlp.dense_4h_to_h.weight.detach().clone() for L in range(n_layers)}

    def quantize_int(W, bits):
        levels = (1 << (bits - 1)) - 1
        scale = W.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / levels
        Wq = torch.round(W / scale).clamp(-levels, levels) * scale
        return Wq

    # ===== Method A: plain INT_residual_bits on full W =====
    print(f"\n[2] plain INT{residual_bits} per-row on full W...", flush=True)
    for L in range(n_layers):
        W = backups[L].to(torch.float32)
        Wq = quantize_int(W, residual_bits)
        model.gpt_neox.layers[L].mlp.dense_4h_to_h.weight.data.copy_(Wq)
    ppl_int = eval_ppl(model, tok, n_eval_tokens)
    print(f"    INT{residual_bits} PPL: {ppl_int:.4f}  (delta {ppl_int - ppl_orig:+.4f})", flush=True)

    # restore
    for L in range(n_layers):
        model.gpt_neox.layers[L].mlp.dense_4h_to_h.weight.data.copy_(backups[L])

    # ===== Method B: hybrid codebook + residual =====
    print(f"\n[3] hybrid: codebook (k={k_sparse}, 8-bit values) + INT{residual_bits} residual...", flush=True)
    hybrid_relerrs = []
    for L in range(n_layers):
        path = hf_hub_download(repo_id="EleutherAI/sae-pythia-410m-65k", filename=f"layers.{L}.mlp/sae.safetensors")
        st = load_file(path)
        W_dec = st["W_dec"]
        if W_dec.shape[1] != d_model: W_dec = W_dec.T
        D = W_dec.to(device=device, dtype=torch.float32).T.contiguous()
        W = backups[L].to(torch.float32)
        # Codebook reconstruction
        Z = batched_mp(D, W, k_sparse)
        # Quantize Z values to 8 bit
        col_max = Z.abs().amax(dim=0, keepdim=True).clamp(min=1e-12)
        q_levels = 127
        Z_q = torch.round(Z / col_max * q_levels).clamp(-q_levels, q_levels) / q_levels * col_max
        W_cb = D @ Z_q
        # Residual
        R = W - W_cb
        R_q = quantize_int(R, residual_bits)
        W_hybrid = W_cb + R_q
        relerr_hybrid = ((W - W_hybrid).norm() / W.norm()).item()
        hybrid_relerrs.append((L, relerr_hybrid))
        model.gpt_neox.layers[L].mlp.dense_4h_to_h.weight.data.copy_(W_hybrid)
        if L % 4 == 0:
            print(f"    L={L}: hybrid relerr={relerr_hybrid:.4f}", flush=True)
    ppl_hybrid = eval_ppl(model, tok, n_eval_tokens)
    print(f"    hybrid PPL: {ppl_hybrid:.4f}  (delta {ppl_hybrid - ppl_orig:+.4f})", flush=True)

    # restore
    for L in range(n_layers):
        model.gpt_neox.layers[L].mlp.dense_4h_to_h.weight.data.copy_(backups[L])

    # ===== Byte accounting =====
    intermediate = model.config.intermediate_size
    n_in = intermediate
    one_W_fp32 = d_model * n_in * 4
    one_W_int = d_model * n_in * residual_bits // 8
    index_bits = 16  # K=65536
    codebook_bytes = (k_sparse * n_in * (index_bits + 8) + 7) // 8 + n_in * 2
    residual_bytes = d_model * n_in * residual_bits // 8
    hybrid_bytes = codebook_bytes + residual_bytes

    print("\n=== BYTE / PPL TRADE-OFF ===", flush=True)
    print(f"  FP32:        {one_W_fp32:>10,} bytes/matrix  PPL {ppl_orig:.2f}", flush=True)
    print(f"  INT{residual_bits}:       {one_W_int:>10,} bytes/matrix  PPL {ppl_int:.2f}", flush=True)
    print(f"  Hybrid:      {hybrid_bytes:>10,} bytes/matrix  PPL {ppl_hybrid:.2f}    "
          f"(={codebook_bytes:,} cb + {residual_bytes:,} resid)", flush=True)
    # Net comparison: hybrid is worth doing only if bytes < INT8 AND PPL is no worse than INT4
    print(f"\nVerdict: hybrid beats plain INT{residual_bits}?  {ppl_hybrid < ppl_int}", flush=True)
    out = {
        "k_sparse": k_sparse, "residual_bits": residual_bits,
        "ppl_orig": ppl_orig, "ppl_int": ppl_int, "ppl_hybrid": ppl_hybrid,
        "bytes_int": one_W_int, "bytes_hybrid": hybrid_bytes,
        "delta_int": ppl_int - ppl_orig, "delta_hybrid": ppl_hybrid - ppl_orig,
    }
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
def main(k_sparse: int = 8, residual_bits: int = 4):
    print(run.remote(k_sparse=k_sparse, residual_bits=residual_bits))
