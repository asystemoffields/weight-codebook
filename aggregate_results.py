"""
Aggregate all probe logs into one clean results report.

Parses results from the log files produced by:
  - probe.py (run3.log)
  - probe_multi.py (probe_multi.log)
  - probe_activation_pca.py (probe_activation_pca.log)
  - probe_pythia.py (probe_pythia.log)
  - probe_cross_layer_pythia.py (probe_cross_layer_pythia.log)
  - probe_compression.py (probe_compression.log)
  - probe_qkv.py (probe_qkv.log)

Writes FINDINGS_final.md with all numbers.
"""
import json, re
from pathlib import Path

ROOT = Path(r"C:\Users\power\documents\weight-codebook")


def extract_last_json(text):
    """Find the last balanced JSON object/array in the text."""
    starts = [(i, ch) for i, ch in enumerate(text) if ch in "[{"]
    for s, _ in reversed(starts):
        for e in range(len(text), s, -1):
            try:
                obj = json.loads(text[s:e])
                return obj
            except Exception:
                continue
    return None


def parse(path: Path):
    if not path.exists(): return None
    text = path.read_text(encoding="utf-8", errors="replace")
    return text


def head(text, n=80):
    if not text: return "(no data)"
    return "\n".join(text.splitlines()[:n])


def report():
    log_files = {
        "codebook_gpt2_layer5": ROOT / "run3.log",
        "codebook_multi_gpt2": ROOT / "probe_multi.log",
        "activation_pca": ROOT / "probe_activation_pca.log",
        "pythia_codebook": ROOT / "probe_pythia.log",
        "pythia_cross_layer": ROOT / "probe_cross_layer_pythia.log",
        "pythia_compression": ROOT / "probe_compression.log",
        "pythia_qkv": ROOT / "probe_qkv.log",
        "virtual_atoms_local": ROOT / "virtual_atoms_modal.log",
    }
    # Extract key numeric findings
    lines = []
    lines.append("# weight-codebook: final autonomous run findings\n\n")
    lines.append("Hardware: Modal A10G (24 GB VRAM). Local laptop has no GPU usable for these probes.\n\n")
    lines.append("## Probe outputs\n\n")
    for name, p in log_files.items():
        lines.append(f"### {name}  (`{p.name}`)\n\n")
        text = parse(p)
        if not text:
            lines.append("- not run / no log\n\n")
            continue
        # Extract summary section if present
        if "=== SUMMARY ===" in text:
            after_summary = text.split("=== SUMMARY ===", 1)[1]
            after_summary = after_summary.split("=== ", 1)[0][:3000]
            lines.append(f"```\n{after_summary.strip()}\n```\n\n")
        elif "=== RESULT ===" in text:
            after = text.split("=== RESULT ===", 1)[1][:3000]
            lines.append(f"```\n{after.strip()}\n```\n\n")
        elif "=== CROSS-LAYER SUMMARY" in text:
            section = text.split("=== CROSS-LAYER SUMMARY", 1)[1][:5000]
            lines.append(f"```\n{section.strip()}\n```\n\n")
        elif "BYTE ACCOUNTING" in text:
            section = text.split("BYTE ACCOUNTING", 1)[1][:3000]
            lines.append(f"```\n{section.strip()}\n```\n\n")
        else:
            lines.append(f"```\n{head(text, 40)}\n...\n```\n\n")
    (ROOT / "FINDINGS_final.md").write_text("".join(lines), encoding="utf-8")
    print(f"wrote {ROOT / 'FINDINGS_final.md'}")


if __name__ == "__main__":
    report()
