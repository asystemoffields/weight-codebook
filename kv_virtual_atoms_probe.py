"""
Virtual KV atoms probe.

Hypothesis: for many attention heads, the past KV cache acts less like a raw
table of n token vectors and more like a small query-response function. If
that's true, there exist m << n virtual atoms (P, U) such that

    softmax(q P^T / sqrt(d)) U  ~=  softmax(q K_old^T / sqrt(d)) V_old

for held-out future queries q. The interesting fingerprint is "preserve what
the next operation needs," not "preserve K and V."

This script answers: for GPT-2-small, how does the required m grow with n,
across layers and heads, and does it beat token-subset baselines?

Outputs:
  - results.csv
  - short_report.md

Designed to run via `modal run kv_virtual_atoms_probe.py` (uses an A10) or
directly via `python kv_virtual_atoms_probe.py --local` on CPU (slower).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------------------
# Hardcoded passages: 4 styles, each long enough to tokenize to >= 320 tokens.
# --------------------------------------------------------------------------------------

PASSAGES = {
    "prose": """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters. "My dear Mr. Bennet," said his lady to him one day, "have you heard that Netherfield Park is let at last?" Mr. Bennet replied that he had not. "But it is," returned she; "for Mrs. Long has just been here, and she told me all about it." Mr. Bennet made no answer. "Do not you want to know who has taken it?" cried his wife impatiently. "You want to tell me, and I have no objection to hearing it." This was invitation enough. "Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a young man of large fortune from the north of England; that he came down on Monday in a chaise and four to see the place, and was so much delighted with it, that he agreed with Mr. Morris immediately; that he is to take possession before Michaelmas, and some of his servants are to be in the house by the end of next week." "What is his name?" "Bingley." "Is he married or single?" "Oh! Single, my dear, to be sure! A single man of large fortune; four or five thousand a year. What a fine thing for our girls!" "How so? how can it affect them?" "My dear Mr. Bennet," replied his wife, "how can you be so tiresome! You must know that I am thinking of his marrying one of them." "Is that his design in settling here?" "Design! Nonsense, how can you talk so! But it is very likely that he may fall in love with one of them, and therefore you must visit him as soon as he comes." "I see no occasion for that. You and the girls may go, or you may send them by themselves, which perhaps will be still better, for as you are as handsome as any of them, Mr. Bingley might like you the best of the party.""",

    "factual": """The mitochondrion is a double-membrane-bound organelle found in most eukaryotic cells. Mitochondria generate most of the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy. The number of mitochondria per cell varies widely; for example, human red blood cells have no mitochondria, whereas liver cells can have more than 2000. The organelle is composed of compartments that carry out specialized functions. These compartments or regions include the outer membrane, the intermembrane space, the inner membrane, and the cristae and matrix. Mitochondrial proteins vary depending on the tissue and the species. In humans, 615 distinct types of proteins have been identified from cardiac mitochondria, whereas in rats, 940 proteins have been reported. The mitochondrial proteome is thought to be dynamically regulated. Although most of a cell's DNA is contained in the cell nucleus, the mitochondrion has its own independent genome that shows substantial similarity to bacterial genomes. Mitochondrial DNA encodes 37 genes including 13 for proteins (polypeptides), 22 for transfer RNA (tRNA) and two for ribosomal RNA (rRNA). The endosymbiotic hypothesis suggests that mitochondria descended from specialized bacteria that survived endocytosis by another species of prokaryote or some other cell, and were incorporated into the cytoplasm. The ability of symbiont bacteria to conduct cellular respiration in host cells that had relied on glycolysis and fermentation would have provided a considerable evolutionary advantage. This symbiotic relationship probably developed 1.7 to 2 billion years ago. A few groups of unicellular eukaryotes have only vestigial mitochondria or derived structures: the microsporidians, metamonads, and archamoebae. These groups appear as the most primitive eukaryotes on phylogenetic trees constructed using rRNA information, which once suggested that they appeared before the origin of mitochondria. However, this is now known to be an artifact of long-branch attraction—they are derived groups and retain genes or organelles derived from mitochondria.""",

    "code": """class LRUCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.cache: dict[int, int] = {}
        self.order: list[int] = []

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            evict = self.order.pop(0)
            del self.cache[evict]
        self.cache[key] = value
        self.order.append(key)


def merge_sort(arr: list[int]) -> list[int]:
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)


def merge(left: list[int], right: list[int]) -> list[int]:
    result: list[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:]); result.extend(right[j:])
    return result


def main() -> None:
    cache = LRUCache(3)
    cache.put(1, 1); cache.put(2, 2); cache.put(3, 3)
    assert cache.get(1) == 1
    cache.put(4, 4)
    assert cache.get(2) == -1
    sample = [5, 2, 8, 1, 9, 3, 7, 4, 6]
    assert merge_sort(sample) == sorted(sample)
    print("ok")


if __name__ == "__main__":
    main()
""",

    "dialogue": """User: I'm trying to figure out how to start. I have a vague idea for a story but every time I sit down to write, I freeze.
Assistant: That's very common. Can you tell me what the idea is, even in one sentence?
User: It's about a woman who finds a letter in her grandmother's attic, and the letter is addressed to a name she's never heard before.
Assistant: Great. That's already a compelling opening. What's stopping you when you sit down?
User: I think I'm worried the writing won't be good enough. I want it to be perfect before I let anyone see it.
Assistant: That's perfectionism. The fix is to write a deliberately rough first draft. Tell yourself, "this draft is allowed to be terrible." What you're protecting against is the empty page, not bad prose.
User: OK. But I also don't know what happens after she finds the letter.
Assistant: You don't need to. Start in the attic. Describe the smell of old paper. Have her open the letter. Read the first line aloud. Then ask: what would she physically do next? Sit down? Call someone? Hide it?
User: I'd want her to hide it, instinctively.
Assistant: Then that's the next sentence. You don't plot a novel before writing it—not on the first try. You discover what happens by following the character's small physical reactions.
User: That feels almost too easy. Don't I need to outline?
Assistant: Some writers outline, some don't. Many discovery writers, like Stephen King, write to find out what happens. If you don't know what happens next, that's not a sign you're a bad writer—it's a sign you need to write the next paragraph.
User: What if it goes nowhere?
Assistant: It probably won't go nowhere if a real character is in a real situation responding honestly. If it does, you cut it and try again. First drafts are practice.
User: All right. I'll start tonight. I'll write the attic scene and stop after the first letter is opened.
Assistant: That's a clean target. Don't reread it tomorrow. Just write the next scene. After three or four scenes you'll have a rhythm, and outlining—if you need it—will be easier because you'll know the character.
""",
}

LAYERS_TO_PROBE = [0, 5, 11]        # first, middle, last of GPT-2 small (12 layers)
HEADS_PER_LAYER = "all"             # "all" or int
N_VALUES = [32, 64, 128, 256]
M_VALUES = [4, 8, 16, 32, 64]
R_FUTURE = 64                        # future query window per (passage, layer)
ADAM_STEPS = 500
ADAM_LR = 5e-2
EARLY_STOP_PATIENCE = 50
EARLY_STOP_DELTA = 1e-5


# --------------------------------------------------------------------------------------
# Core methods
# --------------------------------------------------------------------------------------

def attn_output(q, K, V, scale):
    """q: (r, d), K: (m, d), V: (m, d). Returns (r, d)."""
    import torch
    logits = (q @ K.transpose(-2, -1)) * scale
    attn = torch.softmax(logits, dim=-1)
    return attn @ V


def attn_logsumexp(q, K, scale):
    import torch
    logits = (q @ K.transpose(-2, -1)) * scale
    return torch.logsumexp(logits, dim=-1)


def fit_virtual_atoms(K_old, V_old, Q_train, Y_train, m, *, scale, steps, lr, seed):
    """Learn (P, U) of shape (m, d) each so that softmax(q P^T) U matches Y_train.

    Initialize from a random subset of (K_old, V_old) + small noise (per spec).
    Returns (P, U, train_loss_history).
    """
    import torch
    d = K_old.shape[-1]
    g = torch.Generator(device=K_old.device).manual_seed(seed)
    n = K_old.shape[0]
    idx = torch.randperm(n, generator=g, device=K_old.device)[:m]
    P = (K_old[idx] + 0.01 * torch.randn(m, d, generator=g, device=K_old.device, dtype=K_old.dtype)).clone().detach().requires_grad_(True)
    U = (V_old[idx] + 0.01 * torch.randn(m, d, generator=g, device=K_old.device, dtype=V_old.dtype)).clone().detach().requires_grad_(True)

    opt = torch.optim.Adam([P, U], lr=lr)
    losses = []
    best = float("inf")
    plateau = 0
    for step in range(steps):
        opt.zero_grad()
        Y_hat = attn_output(Q_train, P, U, scale)
        loss = ((Y_hat - Y_train) ** 2).mean()
        loss.backward()
        opt.step()
        l = loss.item()
        losses.append(l)
        if best - l > EARLY_STOP_DELTA:
            best = l; plateau = 0
        else:
            plateau += 1
            if plateau > EARLY_STOP_PATIENCE:
                break
    return P.detach(), U.detach(), losses


def random_subset_baseline(K_old, V_old, m, seed):
    import torch
    g = torch.Generator(device=K_old.device).manual_seed(seed)
    n = K_old.shape[0]
    idx = torch.randperm(n, generator=g, device=K_old.device)[:m]
    return K_old[idx], V_old[idx]


def importance_subset_baseline(K_old, V_old, Q_train, m, scale):
    """Pick m old tokens with highest average attention mass from train queries."""
    import torch
    logits = (Q_train @ K_old.transpose(-2, -1)) * scale     # (r_train, n)
    weights = torch.softmax(logits, dim=-1).mean(dim=0)      # (n,)
    top = torch.topk(weights, k=m).indices
    return K_old[top], V_old[top]


def kmeans_baseline(K_old, V_old, m, scale, *, iters=20, seed=0):
    """Cluster K_old into m clusters; use centers for keys, average V per cluster for values."""
    import torch
    n, d = K_old.shape
    if m >= n:
        return K_old.clone(), V_old.clone()
    g = torch.Generator(device=K_old.device).manual_seed(seed)
    perm = torch.randperm(n, generator=g, device=K_old.device)[:m]
    centers = K_old[perm].clone()
    for _ in range(iters):
        dist = torch.cdist(K_old, centers)
        assign = dist.argmin(dim=-1)
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(m, device=K_old.device, dtype=K_old.dtype)
        new_V = torch.zeros_like(centers)
        for c in range(m):
            mask = assign == c
            cnt = mask.sum()
            if cnt > 0:
                new_centers[c] = K_old[mask].mean(0)
                new_V[c] = V_old[mask].mean(0)
                counts[c] = cnt
            else:
                new_centers[c] = centers[c]
                new_V[c] = V_old[perm[c]]
        centers = new_centers
    return centers, new_V


def rel_mse(Y_hat, Y_true):
    import torch
    num = ((Y_hat - Y_true) ** 2).mean()
    den = (Y_true ** 2).mean().clamp(min=1e-12)
    return (num / den).item()


def mean_cosine(Y_hat, Y_true):
    import torch
    Yh = Y_hat / Y_hat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    Yt = Y_true / Y_true.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (Yh * Yt).sum(-1).mean().item()


# --------------------------------------------------------------------------------------
# Hooking GPT-2 for Q, K, V capture
# --------------------------------------------------------------------------------------

def capture_qkv(model, input_ids):
    """Run a forward pass and capture (Q, K, V) per layer.

    Returns:
        q_all: list of length n_layers, each (n_heads, seq, head_dim)
        k_all: same
        v_all: same
    """
    import torch
    captured: List[Dict[str, "torch.Tensor"]] = []
    handles = []
    cfg = model.config
    n_heads = cfg.n_head
    head_dim = cfg.n_embd // cfg.n_head

    def make_hook(layer_i):
        def hook(module, inp, out):
            # out is the concatenated (q,k,v) tensor of shape (batch, seq, 3*d)
            qkv = out
            batch, seq, _ = qkv.shape
            q, k, v = qkv.split(cfg.n_embd, dim=-1)
            q = q.view(batch, seq, n_heads, head_dim).permute(0, 2, 1, 3)  # (b,h,s,d)
            k = k.view(batch, seq, n_heads, head_dim).permute(0, 2, 1, 3)
            v = v.view(batch, seq, n_heads, head_dim).permute(0, 2, 1, 3)
            captured.append({"q": q[0].detach(), "k": k[0].detach(), "v": v[0].detach()})
        return hook

    for i, block in enumerate(model.transformer.h):
        h = block.attn.c_attn.register_forward_hook(make_hook(i))
        handles.append(h)
    with torch.no_grad():
        model(input_ids)
    for h in handles: h.remove()
    return captured


# --------------------------------------------------------------------------------------
# Main probe
# --------------------------------------------------------------------------------------

@dataclasses.dataclass
class Row:
    passage: str
    layer: int
    head: int
    n: int
    m: int
    method: str
    rel_mse: float
    cosine: float
    train_rel_mse: float
    logsumexp_mae: float
    seed: int


def run_probe(device: str = "cuda") -> List[Row]:
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    torch.set_grad_enabled(False)

    print(f"[probe] device={device}")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    cfg = model.config
    n_heads = cfg.n_head
    head_dim = cfg.n_embd // n_heads
    scale = 1.0 / math.sqrt(head_dim)

    if HEADS_PER_LAYER == "all":
        heads_iter = list(range(n_heads))
    else:
        heads_iter = list(range(int(HEADS_PER_LAYER)))

    rows: List[Row] = []
    for p_name, p_text in PASSAGES.items():
        ids = tok(p_text, return_tensors="pt").input_ids.to(device)
        seq_len = ids.shape[1]
        print(f"[probe] passage='{p_name}' tokens={seq_len}")

        capt = capture_qkv(model, ids)

        for layer_i in LAYERS_TO_PROBE:
            Q_layer = capt[layer_i]["q"]
            K_layer = capt[layer_i]["k"]
            V_layer = capt[layer_i]["v"]

            for head_i in heads_iter:
                Q = Q_layer[head_i].float()      # (seq, d)
                K = K_layer[head_i].float()
                V = V_layer[head_i].float()

                for n in N_VALUES:
                    if n + R_FUTURE > seq_len:
                        continue
                    K_old = K[:n]
                    V_old = V[:n]
                    Q_future = Q[n:n + R_FUTURE]

                    r = Q_future.shape[0]
                    r_train = r // 2
                    Q_train = Q_future[:r_train]
                    Q_test = Q_future[r_train:]

                    # teacher outputs
                    with torch.no_grad():
                        Y_train_true = attn_output(Q_train, K_old, V_old, scale)
                        Y_test_true = attn_output(Q_test, K_old, V_old, scale)
                        lse_test_true = attn_logsumexp(Q_test, K_old, scale)

                    for m in M_VALUES:
                        if m > n:
                            continue
                        for seed in [0, 1, 2]:
                            # -------- virtual atoms --------
                            with torch.enable_grad():
                                P, U, losses = fit_virtual_atoms(
                                    K_old, V_old, Q_train, Y_train_true,
                                    m, scale=scale, steps=ADAM_STEPS, lr=ADAM_LR, seed=seed,
                                )
                            with torch.no_grad():
                                Y_test_hat = attn_output(Q_test, P, U, scale)
                                Y_train_hat = attn_output(Q_train, P, U, scale)
                                lse_test_hat = attn_logsumexp(Q_test, P, scale)
                            rows.append(Row(
                                passage=p_name, layer=layer_i, head=head_i, n=n, m=m,
                                method="virtual_atoms",
                                rel_mse=rel_mse(Y_test_hat, Y_test_true),
                                cosine=mean_cosine(Y_test_hat, Y_test_true),
                                train_rel_mse=rel_mse(Y_train_hat, Y_train_true),
                                logsumexp_mae=(lse_test_hat - lse_test_true).abs().mean().item(),
                                seed=seed,
                            ))

                            # -------- random subset --------
                            Pr, Ur = random_subset_baseline(K_old, V_old, m, seed)
                            with torch.no_grad():
                                Yh = attn_output(Q_test, Pr, Ur, scale)
                                Yh_train = attn_output(Q_train, Pr, Ur, scale)
                                lse = attn_logsumexp(Q_test, Pr, scale)
                            rows.append(Row(
                                passage=p_name, layer=layer_i, head=head_i, n=n, m=m,
                                method="random_subset",
                                rel_mse=rel_mse(Yh, Y_test_true),
                                cosine=mean_cosine(Yh, Y_test_true),
                                train_rel_mse=rel_mse(Yh_train, Y_train_true),
                                logsumexp_mae=(lse - lse_test_true).abs().mean().item(),
                                seed=seed,
                            ))

                            # -------- importance subset --------
                            Pi, Ui = importance_subset_baseline(K_old, V_old, Q_train, m, scale)
                            with torch.no_grad():
                                Yh = attn_output(Q_test, Pi, Ui, scale)
                                Yh_train = attn_output(Q_train, Pi, Ui, scale)
                                lse = attn_logsumexp(Q_test, Pi, scale)
                            rows.append(Row(
                                passage=p_name, layer=layer_i, head=head_i, n=n, m=m,
                                method="importance_subset",
                                rel_mse=rel_mse(Yh, Y_test_true),
                                cosine=mean_cosine(Yh, Y_test_true),
                                train_rel_mse=rel_mse(Yh_train, Y_train_true),
                                logsumexp_mae=(lse - lse_test_true).abs().mean().item(),
                                seed=seed,
                            ))

                            # -------- kmeans --------
                            Pk, Uk = kmeans_baseline(K_old, V_old, m, scale, seed=seed)
                            with torch.no_grad():
                                Yh = attn_output(Q_test, Pk, Uk, scale)
                                Yh_train = attn_output(Q_train, Pk, Uk, scale)
                                lse = attn_logsumexp(Q_test, Pk, scale)
                            rows.append(Row(
                                passage=p_name, layer=layer_i, head=head_i, n=n, m=m,
                                method="kmeans",
                                rel_mse=rel_mse(Yh, Y_test_true),
                                cosine=mean_cosine(Yh, Y_test_true),
                                train_rel_mse=rel_mse(Yh_train, Y_train_true),
                                logsumexp_mae=(lse - lse_test_true).abs().mean().item(),
                                seed=seed,
                            ))
            print(f"[probe]   layer={layer_i} done ({len(rows)} rows total)")
    return rows


def write_csv(rows: List[Row], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["passage","layer","head","n","m","method","rel_mse","cosine","train_rel_mse","logsumexp_mae","seed"])
        for r in rows:
            w.writerow([r.passage, r.layer, r.head, r.n, r.m, r.method,
                        f"{r.rel_mse:.6f}", f"{r.cosine:.6f}", f"{r.train_rel_mse:.6f}",
                        f"{r.logsumexp_mae:.6f}", r.seed])
    print(f"[probe] wrote {path} ({len(rows)} rows)")


def write_report(rows: List[Row], path: Path) -> None:
    """Blunt summary, with tables and a Keep/Mutate/Kill verdict."""
    from collections import defaultdict
    # Aggregate: mean rel_mse over seeds for each (layer, head, n, m, method)
    agg = defaultdict(list)
    for r in rows:
        agg[(r.layer, r.head, r.n, r.m, r.method)].append(r)
    mean_rel = {k: sum(rr.rel_mse for rr in v) / len(v) for k, v in agg.items()}
    mean_train = {k: sum(rr.train_rel_mse for rr in v) / len(v) for k, v in agg.items()}
    mean_lse = {k: sum(rr.logsumexp_mae for rr in v) / len(v) for k, v in agg.items()}

    # Per-(layer, head, n) find smallest m that achieves rel_mse <= threshold for virtual_atoms
    THRESHOLD = 0.05  # 5% rel MSE
    headers = sorted({(k[0], k[1], k[2]) for k in mean_rel.keys()})
    compress_table = []
    for (L, H, n) in headers:
        for method in ["virtual_atoms", "random_subset", "importance_subset", "kmeans"]:
            ms_under = [k[3] for k in mean_rel if k[:3] == (L, H, n) and k[4] == method and mean_rel[k] <= THRESHOLD]
            best_m = min(ms_under) if ms_under else None
            best_relmse = min(mean_rel[k] for k in mean_rel if k[:3] == (L, H, n) and k[4] == method)
            compress_table.append({"layer": L, "head": H, "n": n, "method": method,
                                   "smallest_m_under_5pct": best_m, "best_rel_mse": best_relmse})

    # Count wins of virtual_atoms vs each baseline (at matched m, by lowest mean rel_mse)
    win_counts = {"vs_random": 0, "vs_importance": 0, "vs_kmeans": 0, "tied_or_lost": 0}
    total = 0
    for (L, H, n) in headers:
        for m in M_VALUES:
            key_va = (L, H, n, m, "virtual_atoms")
            if key_va not in mean_rel: continue
            total += 1
            va = mean_rel[key_va]
            r_b = mean_rel.get((L, H, n, m, "random_subset"), float("inf"))
            i_b = mean_rel.get((L, H, n, m, "importance_subset"), float("inf"))
            k_b = mean_rel.get((L, H, n, m, "kmeans"), float("inf"))
            if va < r_b: win_counts["vs_random"] += 1
            if va < i_b: win_counts["vs_importance"] += 1
            if va < k_b: win_counts["vs_kmeans"] += 1
            if not (va < r_b and va < i_b and va < k_b):
                win_counts["tied_or_lost"] += 1

    # Overfit: fraction with large train/test gap for virtual_atoms
    n_overfit = 0; n_total_va = 0
    for k, v in mean_rel.items():
        if k[4] != "virtual_atoms": continue
        n_total_va += 1
        if v - mean_train[k] > 0.1:  # held-out > train by >10pp rel_mse
            n_overfit += 1

    lines = []
    lines.append("# Virtual KV atoms — short report\n")
    lines.append(f"Model: GPT-2 small. Layers probed: {LAYERS_TO_PROBE}. n sweep: {N_VALUES}. m sweep: {M_VALUES}. r_future={R_FUTURE}. seeds=3.\n")
    lines.append(f"Rows total: {len(rows)}. Configs (after seed averaging): {len(mean_rel)}.\n")

    lines.append("\n## 1. Did virtual atoms beat token-subset baselines?\n")
    lines.append(f"- vs random_subset: {win_counts['vs_random']}/{total} configurations\n")
    lines.append(f"- vs importance_subset: {win_counts['vs_importance']}/{total} configurations\n")
    lines.append(f"- vs kmeans: {win_counts['vs_kmeans']}/{total} configurations\n")
    lines.append(f"- virtual_atoms strictly better than all three baselines: {total - win_counts['tied_or_lost']}/{total}\n")

    lines.append("\n## 2. How does required m scale with n (virtual_atoms, threshold rel_mse <= 5%)?\n")
    lines.append("Per (layer, head): smallest m such that mean rel_mse <= 0.05 at each n.\n\n")
    lines.append("| layer | head | n | smallest_m_va | smallest_m_random | smallest_m_importance | smallest_m_kmeans |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    from collections import defaultdict as dd
    per_lhn = dd(dict)
    for row in compress_table:
        per_lhn[(row["layer"], row["head"], row["n"])][row["method"]] = row["smallest_m_under_5pct"]
    # show only first 6 heads of each layer for readability
    for (L, H, n) in headers:
        if H >= 6: continue
        rec = per_lhn.get((L, H, n), {})
        def fmt(x): return "-" if x is None else str(x)
        lines.append(f"| {L} | {H} | {n} | {fmt(rec.get('virtual_atoms'))} | {fmt(rec.get('random_subset'))} | {fmt(rec.get('importance_subset'))} | {fmt(rec.get('kmeans'))} |\n")

    lines.append("\n## 3-4. Most / least compressible (layer, head) pairs\n")
    head_score = dd(list)
    for k, v in mean_rel.items():
        if k[4] != "virtual_atoms": continue
        L, H, n, m, _ = k
        head_score[(L, H)].append(v)
    head_mean = {k: sum(vs)/len(vs) for k, vs in head_score.items()}
    sorted_heads = sorted(head_mean.items(), key=lambda x: x[1])
    lines.append("Most compressible (lowest mean virtual_atoms rel_mse across all n,m):\n")
    for (L, H), v in sorted_heads[:6]:
        lines.append(f"- layer {L} head {H}: mean rel_mse = {v:.4f}\n")
    lines.append("Least compressible (highest):\n")
    for (L, H), v in sorted_heads[-6:]:
        lines.append(f"- layer {L} head {H}: mean rel_mse = {v:.4f}\n")

    lines.append("\n## 5. Overfitting?\n")
    lines.append(f"virtual_atoms configs with held-out > train by >10pp rel_mse: {n_overfit}/{n_total_va}\n")

    # Verdict
    win_rate = (total - win_counts["tied_or_lost"]) / max(1, total)
    # is m growing slowly with n? for virtual_atoms, take median smallest_m across heads, per n
    by_n = dd(list)
    for row in compress_table:
        if row["method"] != "virtual_atoms" or row["smallest_m_under_5pct"] is None: continue
        by_n[row["n"]].append(row["smallest_m_under_5pct"])
    med_m_per_n = {n: sorted(v)[len(v)//2] for n, v in by_n.items() if v}
    lines.append("\n## 6. Median smallest_m for virtual_atoms (5% rel_mse threshold) by n\n")
    for n in sorted(med_m_per_n):
        lines.append(f"- n={n}: median smallest_m = {med_m_per_n[n]}\n")

    growth_slow = False
    if len(med_m_per_n) >= 2:
        ns = sorted(med_m_per_n)
        m_ratio = med_m_per_n[ns[-1]] / max(1, med_m_per_n[ns[0]])
        n_ratio = ns[-1] / ns[0]
        growth_slow = m_ratio < n_ratio * 0.5
        lines.append(f"\nm grows by factor {m_ratio:.2f} while n grows by {n_ratio:.2f}; "
                     f"{'SUBLINEAR' if growth_slow else 'roughly LINEAR or worse'}.\n")

    if win_rate >= 0.5 and growth_slow and n_overfit / max(1, n_total_va) < 0.3:
        verdict = "KEEP"
    elif win_rate >= 0.3 or growth_slow:
        verdict = "MUTATE"
    else:
        verdict = "KILL"
    lines.append(f"\n## VERDICT: **{verdict}**\n")
    lines.append(f"- win_rate (virtual_atoms strictly best): {win_rate:.2f}\n")
    lines.append(f"- m grows sublinearly with n: {growth_slow}\n")
    lines.append(f"- overfit rate: {n_overfit / max(1, n_total_va):.2f}\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[probe] wrote {path}")


# --------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------

def _run_and_save(device: str, out_dir: Path):
    rows = run_probe(device=device)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / "results.csv")
    write_report(rows, out_dir / "short_report.md")


def local_main():
    import torch
    device = "cuda" if (sys.argv and "--cuda" in sys.argv) else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    out_dir = Path(__file__).parent / "virtual_atoms_results"
    _run_and_save(device, out_dir)


# Modal section -------------------------------------------------------------------------
try:
    import modal  # noqa: F401
    HAVE_MODAL = True
except Exception:
    HAVE_MODAL = False


if HAVE_MODAL and __name__ != "__main__":
    pass  # don't auto-build app when imported by another script


if HAVE_MODAL:
    import modal as _modal

    _image = (
        _modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "numpy<2.0",
            "torch==2.5.1",
            "transformers==4.46.3",
            "huggingface_hub[hf_transfer]>=0.26,<1.0",
        )
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
    )
    _vol = _modal.Volume.from_name("hf-cache-weight-codebook", create_if_missing=True)
    _app = _modal.App("kv-virtual-atoms-probe")

    @_app.function(
        image=_image, gpu="A10G", timeout=2400,
        volumes={"/root/.cache/huggingface": _vol},
    )
    def remote_probe():
        from pathlib import Path as _P
        rows = run_probe(device="cuda")
        out = _P("/results"); out.mkdir(exist_ok=True)
        write_csv(rows, out / "results.csv")
        write_report(rows, out / "short_report.md")
        return {
            "n_rows": len(rows),
            "csv": (out / "results.csv").read_text(),
            "report": (out / "short_report.md").read_text(),
        }

    @_app.local_entrypoint()
    def main():
        result = remote_probe.remote()
        out_dir = Path(__file__).parent / "virtual_atoms_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.csv").write_text(result["csv"])
        (out_dir / "short_report.md").write_text(result["report"])
        print(f"wrote {result['n_rows']} rows")
        print("\n--- REPORT ---\n")
        print(result["report"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run locally (CPU or CUDA if available)")
    parser.add_argument("--cuda", action="store_true", help="Force CUDA when running locally")
    args, _ = parser.parse_known_args()
    if args.local:
        local_main()
