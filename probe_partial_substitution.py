"""
Partial-substitution probe.

End-to-end codebook substitution exploded PPL (19.5 -> 81586). But maybe
partial substitution (a subset of layers) works fine. This probe sweeps
the number of substituted layers and reports PPL.

If only-shallow-layers or only-deep-layers tolerate codebook substitution
while the rest stays FP32, we could compress *part* of the model and avoid
catastrophic collapse.
"""
import modal

app = modal.App("probe-partial-sub")

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
def run(k_sparse: int = 32, n_eval_tokens: int = 4096):
    import json, time, math
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from datasets import load_dataset

    device = "cuda"
    torch.set_grad_enabled(False)
    print(f"=== Partial-substitution probe (Pythia 410M) ===  k_sparse={k_sparse}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-410m", torch_dtype=torch.float32, device_map="cuda",
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers

    backups = {L: model.gpt_neox.layers[L].mlp.dense_4h_to_h.weight.detach().clone() for L in range(n_layers)}

    # Pre-compute codebook reconstruction for every layer
    print("[1] computing codebook reconstructions for all layers...", flush=True)
    recons = {}
    relerrs = {}
    for L in range(n_layers):
        path = hf_hub_download(repo_id="EleutherAI/sae-pythia-410m-65k", filename=f"layers.{L}.mlp/sae.safetensors")
        st = load_file(path)
        W_dec = st["W_dec"]
        if W_dec.shape[1] != d_model: W_dec = W_dec.T
        D = W_dec.to(device=device, dtype=torch.float32).T.contiguous()
        W = backups[L].to(torch.float32)
        Z = batched_mp(D, W, k_sparse)
        W_recon = D @ Z
        recons[L] = W_recon
        relerrs[L] = ((W - W_recon).norm() / W.norm()).item()
        if L % 4 == 0: print(f"  L={L} relerr={relerrs[L]:.3f}", flush=True)

    ppl_orig = eval_ppl(model, tok, n_eval_tokens)
    print(f"\n[2] baseline PPL: {ppl_orig:.4f}", flush=True)

    # ----- Sweep: substitute first K layers -----
    print("\n[3] substituting first K layers and measuring PPL...", flush=True)
    rows = []
    for K in [0, 1, 2, 4, 8, 12, 16, 20, 24]:
        # Reset all
        for L in range(n_layers):
            model.gpt_neox.layers[L].mlp.dense_4h_to_h.weight.data.copy_(backups[L])
        # Substitute first K
        for L in range(K):
            model.gpt_neox.layers[L].mlp.dense_4h_to_h.weight.data.copy_(recons[L])
        ppl = eval_ppl(model, tok, n_eval_tokens)
        rows.append({"first_K": K, "ppl": ppl, "delta": ppl - ppl_orig})
        print(f"  first {K:>2} layers: PPL={ppl:.4f}  delta={ppl-ppl_orig:+.4f}", flush=True)

    # Reset
    for L in range(n_layers): model.gpt_neox.layers[L].mlp.dense_4h_to_h.weight.data.copy_(backups[L])

    # ----- Sweep: substitute last K layers -----
    print("\n[4] substituting last K layers and measuring PPL...", flush=True)
    for K in [0, 1, 2, 4, 8, 12, 16, 20, 24]:
        for L in range(n_layers): model.gpt_neox.layers[L].mlp.dense_4h_to_h.weight.data.copy_(backups[L])
        for L in range(n_layers - K, n_layers):
            model.gpt_neox.layers[L].mlp.dense_4h_to_h.weight.data.copy_(recons[L])
        ppl = eval_ppl(model, tok, n_eval_tokens)
        rows.append({"last_K": K, "ppl": ppl, "delta": ppl - ppl_orig})
        print(f"  last  {K:>2} layers: PPL={ppl:.4f}  delta={ppl-ppl_orig:+.4f}", flush=True)

    print("\n=== Per-layer relerrs (just for reference) ===", flush=True)
    for L in range(n_layers):
        print(f"  L={L:>2}: relerr={relerrs[L]:.3f}", flush=True)

    print(json.dumps(rows, indent=2), flush=True)
    return rows


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
def main(k_sparse: int = 32):
    print(run.remote(k_sparse=k_sparse))
