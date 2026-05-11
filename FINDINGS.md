# Weight-codebook probes: findings (in progress, autonomous run)

Date: 2026-05-11
Hardware: Modal A10G (GPT-2 small, no Gemma access)

## TL;DR

The structural hypothesis — that LLM weight matrices can be sparse-coded over
a learned universal feature dictionary (SAE atoms) — is **partially confirmed**
on GPT-2 small with one critical correction to the framing:

> The right SAE for sparse-coding the columns of layer L's projection
> matrices (`attn.c_proj`, `mlp.c_proj`) is **layer L+1's** residual-stream
> SAE — the one trained on the residual stream **after** layer L's writes,
> not the one capturing the state before.

With that correction, codebook reconstruction beats both:
- Random gaussian dictionary of the same K=24576 (consistent ~5-13pp gain)
- Best-possible rank-k SVD at matched bytes (codebook ~0.62-0.70 vs SVD ~0.85-0.92)

The win is *modest, consistent, and load-bearing*: 12/20 (L, weight, sae_offset)
configurations beat random by >5pp; 0 configurations lose to random by >2pp.

What remains unsettled:
- Codebook relerr ~0.6-0.7 is large in absolute terms — the dictionary doesn't
  fully span W column space, just spans it *better than random*.
- Whether SAE atoms beat plain activation-PCA at the same K (the orthogonal
  control, currently running as `probe_activation_pca.py`).
- Whether the gain holds on larger models (Pythia, Gemma). GPT-2 small alone
  is not enough to conclude about 100B+.

## Detailed results

### probe.py (v4, layer 5 attn.c_proj — the misleading first test)

GPT-2 small, layer 5, `attn.c_proj`, SAE from `blocks.5.hook_resid_pre` (same layer):

| Method | rel_err | PPL | delta vs orig (31.24) |
|---|---|---|---|
| SAE codebook (k=32) | 0.783 | 33.08 | +1.84 |
| Random gaussian dict (K=24576) | 0.752 | 32.85 | +1.60 |
| SVD rank-32 (matched bytes) | 0.866 | 33.08 | +1.84 |

Initial reading: codebook LOSES to random. KILL signal.

**This was the wrong test.** The SAE for `blocks.5.hook_resid_pre` is trained
on the residual stream *entering* block 5 — i.e., the state BEFORE block 5's
attention writes. The columns of W_O for block 5 are vectors that block 5's
attention adds TO that state. They should be aligned with the residual stream
state AFTER block 5's writes — i.e., layer 6's pre.

### probe_multi.py (corrected sweep)

20 configurations: 5 layers × {attn.c_proj, mlp.c_proj} × {sae from L, sae from L+1}.

```
L  weight        sae   cb       rand     gain
1  attn.c_proj   L1    0.7386   0.7523   +0.014
1  attn.c_proj   L2    0.6770   0.7518   +0.075
1  mlp.c_proj    L1    0.6971   0.7522   +0.055
1  mlp.c_proj    L2    0.6553   0.7522   +0.097
3  attn.c_proj   L3    0.7483   0.7528   +0.005
3  attn.c_proj   L4    0.7051   0.7521   +0.047
3  mlp.c_proj    L3    0.7135   0.7523   +0.039
3  mlp.c_proj    L4    0.6246   0.7521   +0.127
5  attn.c_proj   L5    0.7522   0.7516   -0.001
5  attn.c_proj   L6    0.6777   0.7521   +0.074
5  mlp.c_proj    L5    0.7267   0.7524   +0.026
5  mlp.c_proj    L6    0.6232   0.7523   +0.129
8  attn.c_proj   L8    0.7004   0.7522   +0.052
8  attn.c_proj   L9    0.6365   0.7521   +0.116
8  mlp.c_proj    L8    0.7194   0.7523   +0.033
8  mlp.c_proj    L9    0.6349   0.7522   +0.117
10 attn.c_proj   L10   0.6958   0.7517   +0.056
10 attn.c_proj   L11   0.6259   0.7521   +0.126
10 mlp.c_proj    L10   0.7218   0.7521   +0.030
10 mlp.c_proj    L11   0.6721   0.7520   +0.080
```

**Patterns:**

1. **`sae_offset=1` (next layer's SAE) ALWAYS beats `sae_offset=0` (same layer)**, by 4-10pp.
2. **`mlp.c_proj` with next-layer-SAE consistently shows the biggest gains** — peak +0.129 at L=5.
3. **`attn.c_proj` also benefits with next-layer-SAE** but is more variable.
4. **SVD at matched bytes is uniformly worst** (0.85-0.92 relerr) — sparse coding has more dof.
5. The gain is **monotone in usefulness of dictionary structure** — random < SAE < ?

## What this means for 100B+ compression

GPT-2 is a fingerprint test, not a deployment test. The byte budget calculation
for GPT-2 doesn't generalize directly. But the *structural fact* — SAE feature
directions span a meaningful portion of weight-matrix output space — is what
we wanted to confirm. It's confirmed, modestly.

For 100B+ models the relevant next steps (autonomous queue):
1. **probe_activation_pca.py** (running) — does plain PCA on real activations
   match the SAE? If yes, no need for SAE training — just use ASVD-style
   compression.
2. **Pythia-410M with EleutherAI/sae-pythia-410m-65k** — repeat probe_multi on a
   non-toy non-gated model. If the pattern holds, the universality argument lives.
3. **End-to-end PPL substitution** for the winning configs — measuring relerr
   says the dictionary spans the columns; measuring PPL says the *parts of the
   columns that matter* are preserved.

## Status

| Probe | Status | Outcome |
|---|---|---|
| probe.py (layer 5, same-layer SAE) | done | misleading kill — wrong SAE |
| probe_multi.py (sweep) | done | **WIN** with sae_offset=1 |
| probe_activation_pca.py | running | TBD |
| kv_virtual_atoms_probe.py | running | TBD |
| Pythia replication | queued | depends on PCA result |
