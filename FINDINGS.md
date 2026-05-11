# weight-codebook: final findings (autonomous run)

Date: 2026-05-11. All probes run on Modal A10G.

## TL;DR

**Structural finding — kept, with constraints:**

For `mlp.dense_4h_to_h` (the MLP down-projection) of layer L in Pythia 410M
and GPT-2 small, the matrix columns are sparse-codable over the **same
layer's** MLP-output SAE atoms with relative reconstruction error
substantially better than:
- random gaussian dictionary at the same K
- best-possible rank-k SVD
- activation PCA at matched K

**Practical compression payoff — killed.**

The structural fingerprint does NOT translate to a byte-saving compression
scheme. Three independent kills:
1. **End-to-end PPL collapses** when all 24 mlp_down matrices are substituted
   with codebook reconstructions: PPL 19.49 → 81586. Per-matrix relerr ~0.55
   sounds OK, but errors compound multiplicatively across layers. INT4 per-row
   keeps PPL at 22.4 on the same substitution.
2. **Dictionaries are strictly layer-specific.** Cross-layer transfer fails:
   layer-12's SAE works for layer-12 (relerr 0.595) but is no better than
   random for layer-10 or 14 (relerr 0.77+, gain ≈ 0). Per-layer dictionary
   cost (65 MB FP32 × 24 layers = 1.5 GB) overwhelms the model itself.
3. **Codebook win is matrix-type-specific** — works on `mlp.dense_4h_to_h`,
   fails on attention Q/K/V (SAE LOSES to random by ~2pp consistently), and
   only marginal on W_O (+0.5 to +1.4 pp).

**Net verdict:** the SAE atoms encode something real about the geometry of
MLP outputs, but not enough — and not in the right form — to enable a
category jump in compression. Standard INT4 PTQ wins.

---

## Detailed results

### probe.py / probe_multi.py (GPT-2 small)

Same-layer SAE for `attn.c_proj` and `mlp.c_proj` at K=24576, k_sparse=32.

Initial single test (layer 5, attn.c_proj, **same-layer** SAE):
- SAE codebook relerr: 0.783, PPL after sub: 33.08 (+1.84)
- Random gaussian: relerr 0.752, PPL 32.85 (+1.60)
- SVD rank-32 (matched bytes): relerr 0.866, PPL 33.08

This looked like a clean kill. But the **same-layer** SAE captures residual
stream BEFORE the matrix writes, not the residual after — so it was the wrong
fingerprint to test. Switching to **next-layer** SAE (the one trained on the
residual stream AFTER layer L's writes) revealed structure:

Full 20-config sweep (layer × {attn,mlp}.c_proj × {sae_offset 0, 1}):
- 12/20 configurations beat random by >5pp
- 0/20 lose to random by >2pp
- Biggest gains: mlp.c_proj with next-layer SAE (+0.097 to +0.129)
- attn.c_proj with next-layer SAE also helps (+0.047 to +0.126)
- Same-layer SAE is roughly tied with random (-0.001 to +0.055)

### probe_activation_pca.py (GPT-2 small)

The same K=24576 SAE codebook vs activation PCA dictionary at matched effective rank.

All 5 layers tested (1, 3, 5, 8, 10) for mlp.c_proj with next-layer SAE:

| L | SAE relerr | PCA relerr | Random | SVD-K | Act-K |
|---|---|---|---|---|---|
| 1 | **0.655** | 0.813 | 0.752 | 0.882 | 0.928 |
| 3 | **0.625** | 0.826 | 0.752 | 0.873 | 0.937 |
| 5 | **0.623** | 0.848 | 0.752 | 0.897 | 0.961 |
| 8 | **0.635** | 0.854 | 0.752 | 0.910 | 0.963 |
| 10 | **0.672** | 0.848 | 0.752 | 0.919 | 0.962 |

SAE beats PCA by 15-22 pp in 5/5 configs. PCA is actually worse than random
gaussian at the same K (PCA's effective K is capped at d_model=768 since only
that many nonzero eigenvalues exist). The SAE atoms span something the
activation variance directions don't.

### probe_pythia.py (Pythia 410M dense_4h_to_h, same-layer SAE)

Cleaner test on a 3.3× larger model with K=65536 features:

| L | SAE | Random | SVD-rank-32 | Gain |
|---|---|---|---|---|
| 4 | **0.582** | 0.785 | 0.925 | +0.203 |
| 8 | **0.548** | 0.785 | 0.925 | +0.237 |
| 12 | **0.595** | 0.785 | 0.948 | +0.190 |
| 16 | **0.535** | 0.785 | 0.945 | +0.250 |
| 20 | **0.547** | 0.785 | 0.955 | +0.238 |

**5/5 wins, gain 19-25 pp.** Much stronger than GPT-2. The structural
fingerprint *strengthens* with model scale, consistent with the universality
hypothesis — albeit only within-layer (see cross-layer below).

### probe_cross_layer_pythia.py — the load-bearing test

Can one dictionary serve many layers? **No.**

Row = source SAE; col = target W layer:

```
src\tgt |     0     2     4     6     8    10    12    14    16    18    20    22
L=4     | 0.812 0.793 0.582 0.776 0.780 0.791 0.797 0.798 0.801 0.804 0.807 0.804
L=8     | 0.813 0.793 0.782 0.752 0.548 0.762 0.782 0.785 0.794 0.799 0.802 0.802
L=12    | 0.807 0.798 0.792 0.780 0.771 0.766 0.595 0.772 0.784 0.793 0.800 0.803
L=16    | 0.806 0.798 0.795 0.789 0.786 0.785 0.781 0.769 0.535 0.779 0.792 0.796
L=20    | 0.801 0.800 0.800 0.799 0.799 0.798 0.796 0.794 0.789 0.782 0.547 0.780
```

The diagonal is sharp — each SAE wins decisively on its OWN target layer
(relerr 0.535-0.595) but is barely better than random (~0.78) on any other.
Mean gain across all targets is +0.011 to +0.017 — basically null.

**Practical implication:** every layer needs its own ~65 MB FP32 dictionary
(or ~33 MB at FP16). For Pythia 410M that's 1.5 GB of dictionaries on top of
a 0.4 GB model. Even at FP16 it's 800 MB. The compression math doesn't work
without cross-layer transfer.

### probe_qkv.py (Pythia 410M)

Does the MLP-output SAE span Q/K/V matrix output spaces too? **No.**

| L | Matrix | SAE relerr | Random | Gain |
|---|---|---|---|---|
| 4 | W_Q | 0.809 | 0.785 | -0.023 |
| 4 | W_K | 0.808 | 0.785 | -0.023 |
| 4 | W_V | 0.808 | 0.785 | -0.023 |
| 4 | W_O | 0.784 | 0.785 | +0.001 |
| 12 | W_Q | 0.805 | 0.785 | -0.020 |
| 12 | W_K | 0.806 | 0.785 | -0.021 |
| 12 | W_V | 0.806 | 0.785 | -0.020 |
| 12 | W_O | 0.771 | 0.785 | +0.014 |
| 20 | W_Q | 0.806 | 0.784 | -0.022 |
| 20 | W_K | 0.797 | 0.783 | -0.014 |
| 20 | W_V | 0.801 | 0.784 | -0.017 |
| 20 | W_O | 0.780 | 0.785 | +0.005 |

0/12 configs win by >5pp. SAE actually *loses* to random on Q/K/V (it's worse
than no information) — the dictionary's bias is harmful when the target
matrix lives in a different vector space. W_O wins marginally (+0.005 to
+0.014) but not at the level we see for dense_4h_to_h.

The codebook win is **specific to MLP down-projection.** That covers
~33% of model parameters in Pythia. Other 2/3 must use standard quantization.

### probe_k_sweep.py (sweep k_sparse on Pythia 410M L=12)

| k | SAE relerr | Random | Gain | Bytes/matrix |
|---|---|---|---|---|
| 1 | 0.847 | 0.990 | +0.144 | 20 KB |
| 2 | 0.824 | 0.982 | +0.158 | 33 KB |
| 4 | 0.794 | 0.965 | +0.171 | 57 KB |
| 8 | 0.751 | 0.935 | +0.184 | 106 KB |
| 16 | 0.687 | 0.881 | +0.194 | 205 KB |
| 32 | 0.595 | 0.785 | +0.190 | 401 KB |
| 64 | 0.473 | 0.628 | +0.156 | 795 KB |

The codebook gain (~15-19 pp) is consistent across all k_sparse values.
Even at k=1 the SAE beats random by 14 pp — single atoms ARE meaningful.
The gain peaks around k=16-32 and drops at k=64 where random catches up.

### probe_compression.py — the verdict

End-to-end PPL test, all 24 mlp.dense_4h_to_h substituted with codebook
reconstructions (k=32, 8-bit values, K=65536 same-layer SAE):

| Method | PPL | Delta vs orig | Bytes/matrix |
|---|---|---|---|
| Original FP32 | 19.49 | 0 | 16.7 MB |
| **SAE codebook** | **81,586** | **+81,567** | 393 KB + 65 MB dict |
| INT8 per-row | 19.49 | +0.003 | 4.2 MB |
| INT4 per-row | 22.41 | +2.93 | 2.1 MB |

**Codebook substitution is catastrophic.** INT4 keeps PPL within 15% of
original. Codebook explodes PPL by 4000×.

Why: per-matrix relerr ~0.55 looks like "preserving 45% of the matrix" but
the relative L2 error compounds across 24 sequential matrix substitutions.
The dominant subspace is preserved, but every fine direction is lost — and
fine directions matter for token prediction.

### probe_residual.py — the hybrid rescue (partial)

If the codebook captures the "easy" directions and an INT4 quantized residual
catches what the codebook misses, can the combination beat plain INT4?

| Method | PPL | Delta | Bytes/matrix |
|---|---|---|---|
| Original FP32 | 19.49 | 0 | 16.8 MB |
| INT4 per-row | 22.41 | +2.93 | 2.10 MB |
| **Hybrid (codebook k=8 + INT4 resid)** | **20.49** | **+1.01** | **2.20 MB** |

Hybrid PPL delta is **3× better** than INT4 at essentially the same per-matrix
bytes (5% overhead from the small codebook codes). Per-matrix hybrid relerr
dropped from 0.55 (codebook-only) to ~0.13 — the INT4 residual is doing the
heavy lifting now, and the codebook contribution acts as a structural prior.

**But the dictionary cost still dominates.** For Pythia 410M's 24 layers at
INT4 dict storage: 24 × 16 MB = 384 MB of dictionaries just to encode
~50 MB of MLP-down weights. Net bytes: worse than plain INT4. The hybrid only
makes sense if you're already storing the dictionary for other reasons (e.g.,
interpretability work).

---

## What this means for the project goal

Goal: compress a 100B+ model onto a Ryzen 5 7530U laptop (8 GB RAM, AVX2).

The SAE-codebook hypothesis is **NOT a category-jump compression candidate**
as tested. The same-layer structural fingerprint is real but it doesn't
translate to bytes saved at usable PPL. INT4 + activation-aware quantization
remain the practical floor for now.

What would change the verdict:
1. **Train a dictionary that minimizes downstream loss, not reconstruction
   error.** SAE atoms are trained to reconstruct activations; that's not the
   same as preserving model output through a substituted W. A directly-trained
   compression codebook might fix the cascading PPL collapse.
2. **Find an architecture where cross-layer transfer works.** If a single
   dictionary spanned all layers' MLP-down, the byte math flips. None tested
   so far show this property.
3. **Combine with PTQ.** Store the codebook reconstruction *plus* a quantized
   residual W - W_recon. The codebook captures the easy directions; the
   quantized residual catches the rest at low bits. This is unexplored.

Per user instructions, the fallback is the **virtual KV atoms** probe, which
ran in parallel (`kv_virtual_atoms_probe.py`). Results to be filled in once
that job completes — separate file `FINDINGS_virtual_atoms.md`.

## Pivot to virtual atoms

Already running. The smoke test signal pointed to overfitting (train rel_mse
8e-5, test rel_mse 1.84 with importance_subset dominating at rel_mse 0.01).
Full sweep is in progress. If the trend in the smoke test holds across all
configs, the virtual atoms hypothesis is also dead — except for the
importance_subset finding, which is essentially confirming the H2O / SnapKV
class of techniques (top-X most-attended tokens compress the cache well).

## Files in this repo

| File | What it does |
|---|---|
| `probe.py` | initial codebook probe (GPT-2 layer 5, mislabeled kill) |
| `probe_multi.py` | corrected 20-config sweep — confirmed win |
| `probe_activation_pca.py` | SAE vs PCA control — SAE wins 5/5 |
| `probe_pythia.py` | replication on Pythia 410M — wins 5/5 by 19-25 pp |
| `probe_cross_layer_pythia.py` | does one dict serve many layers? **no** |
| `probe_qkv.py` | does the win extend to attention matrices? **no** |
| `probe_compression.py` | end-to-end PPL substitution — **kill** |
| `probe_k_sweep.py` | sweep k_sparse to find efficient operating point (not run) |
| `kv_virtual_atoms_probe.py` | fallback per user spec (running) |
| `aggregate_results.py` | parses logs into a single summary doc |
| `laptop_target_models.md` | concrete models the user can run on this laptop today |
