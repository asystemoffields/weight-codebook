# One-shot: create the public GitHub repo and push everything.
# Run after FINDINGS.md is finalized.
$ErrorActionPreference = "Stop"
Set-Location "C:\Users\power\documents\weight-codebook"

# Stage everything
git add .
git -c user.name="Claude" -c user.email="invertedinkuniverse@proton.me" commit -m "final autonomous-run snapshot" --allow-empty

# Create repo and push
gh repo create weight-codebook --public --description "SAE-codebook factorization of LLM weight matrices: empirical probes on GPT-2 small and Pythia 410M" --source=. --remote=origin --push

Write-Host "Done. URL:"
gh repo view weight-codebook --json url --jq .url
