# weight-codebook

Empirical probes into whether LLMs are compressible by exploiting representation
geometry rather than numerical statistics.

## Probes

### `probe.py` — Universal feature codebook for weights

Tests whether the columns of `W_O` (attention output projection) in a Gemma 2 9B
layer can be sparse-coded over that layer's Gemma Scope SAE decoder atoms.

If `||W - D @ Z||_F` is small with `Z` sparse (and `D` shared across many things),
then weight matrices are factorable over a learned universal basis — predicting
~5-10x compression beyond standard 4-bit quantization.

Baselines: SVD rank-K at matched bytes, random gaussian dictionary at same K.

Run:
```
modal run probe.py
```

### `kv_virtual_atoms_probe.py` — KV cache as a query→response function

Tests whether `softmax(q K_old^T) V_old` for held-out future queries can be
approximated by `softmax(q P^T) U` with `m << n` virtual atoms. The fingerprint
of interest is *not* low K/V reconstruction error — it's preservation of the
attention output that the next operation actually consumes.

Sweeps GPT-2 small across layers, heads, prefix lengths n ∈ {32..256}, and
atom counts m ∈ {4..64}. Compares virtual atoms (learned P, U via Adam) to
random subset, importance subset, and k-means baselines.

Outputs `results.csv` and `short_report.md` with a Keep / Mutate / Kill verdict.

Run:
```
modal run kv_virtual_atoms_probe.py
```

Or locally (CPU is fine for GPT-2 small):
```
python kv_virtual_atoms_probe.py --local
```
