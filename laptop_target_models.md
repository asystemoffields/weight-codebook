# Concrete models to try on this laptop right now

Hardware: Ryzen 5 7530U (Zen 3, AVX2 only) + 8 GB DDR4-3200 RAM + AMD Radeon Vega 7 iGPU (no useful VRAM) + 512 GB KIOXIA NVMe (~3 GB/s).

The bandwidth wall: each decode token must read every active weight from RAM. With ~18 GB/s effective RAM bandwidth, the throughput ceiling is:

```
tok/s_ceiling ≈ ram_bandwidth_GBps / (active_params_GB)
```

So if your active-weight working set is `X` GB, you cannot exceed `18/X` tok/s no matter how fast the CPU is. For 100B+ targets, the only models that fit this laptop's `tok/s ≥ 1` constraint are MoE with small active sets, *and* even then storage capacity (~7 GB RAM, 232 GB free disk) forces NVMe streaming.

## Tier 1 — runs comfortably in RAM (recommended for daily use)

These fit comfortably in ~7 GB RAM with headroom for KV cache:

| Model | File | Quant | Size | Est. tok/s | Notes |
|---|---|---|---|---|---|
| Qwen3-4B-Instruct | `Qwen3-4B-Instruct-IQ4_XS.gguf` | IQ4_XS | ~2.5 GB | ~6 (verified by user) | Current baseline |
| Qwen3-4B-Thinking | `Qwen3-4B-Thinking-IQ4_XS.gguf` | IQ4_XS | ~2.5 GB | ~6 | Reasoning variant |
| DeepSeek-R1-Distill-Qwen-7B | `DeepSeek-R1-Distill-Qwen-7B-IQ3_M.gguf` | IQ3_M | ~3.4 GB | ~3-4 | **~55% AIME 2024**, distilled from R1 671B |
| DeepSeek-R1-Distill-Qwen-7B | `DeepSeek-R1-Distill-Qwen-7B-IQ4_XS.gguf` | IQ4_XS | ~4.1 GB | ~3-4 | Better quality if RAM allows |
| BitNet b1.58 2B4T | `bitnet-b1.58-2B-4T-q1_0.gguf` | native ternary | ~1.2 GB | ~10-20 (T-MAC kernel) | Microsoft, only native-ternary model at scale |
| Phi-4-mini-reasoning | `Phi-4-mini-reasoning-Q4_K_M.gguf` | Q4_K_M | ~2.4 GB | ~4-5 | Strong math under 4B |

## Tier 2 — tight fit, NVMe will thrash but works

These exceed RAM slightly; ik_llama.cpp's mmap will stream from NVMe. First token may be slow, then steady-state speed depends on which experts/layers are touched.

| Model | File | Quant | Size | Est. tok/s | Notes |
|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-14B | `IQ3_XXS` | ~5.4 GB | ~1.5-2 | **~70% AIME 2024** |
| Qwen3-30B-A3B | `IQ2_XXS` | ~8.5 GB | ~2-3 | MoE: 3B active, ~70% on benchmarks |
| Phi-4-reasoning (14B) | `Q4_K_S` | ~7.6 GB | ~1-2 | o3-mini-style reasoning traces |

## Tier 3 — does not fit; literature target unachievable

These would require >32 GB RAM at meaningful quants. Listed only to be explicit about what the laptop *cannot* run:

| Model | Smallest viable quant | RAM needed | Why it can't work here |
|---|---|---|---|
| DeepSeek-V3 671B | IQ1_M | 169 GB | RAM short by 23x; NVMe bandwidth ceiling ~0.4 tok/s |
| Qwen3-235B-A22B | IQ2_XXS | 60 GB | RAM short by 8x; ~0.7 tok/s ceiling |
| Kimi K2 1T | IQ1_M | 200 GB | Same as above |
| Llama-3.1-70B | IQ2_XXS | 20 GB | RAM short by 2.5x; would thrash |

## Practical recommendation

Switch to **DeepSeek-R1-Distill-Qwen-7B at IQ3_M (~3.4 GB)** for reasoning tasks — gets you most of the way to a "100B-derived capability" experience on this hardware. It is the literal, currently shipped, "compressed-from-671B" artifact that fits comfortably in RAM with room for 8k+ context.

For the highest absolute quality at slight tok/s cost, **Qwen3-30B-A3B at IQ2_XXS** is the most ambitious model that runs at all on this hardware (MoE saves us — only 3B active params per token, ~3-4× the active-weight bandwidth bill of Qwen3-4B).

The category jump beyond Tier 1 requires research-grade compression that doesn't ship as a public artifact yet.
