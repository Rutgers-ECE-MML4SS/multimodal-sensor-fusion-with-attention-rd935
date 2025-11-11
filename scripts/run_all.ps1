# ============================================================
# Multimodal Sensor Fusion - Full Run
# Runs Early, Late, and Hybrid fusion sequentially using base.yaml
# ============================================================

$ErrorActionPreference = "Stop"
$python = "python"

# --- 1. EARLY ---
Write-Host "=== [1/3] Running EARLY fusion (GPU) ==="
& $python src/train.py `
  model.fusion_type=early
if ($LASTEXITCODE -ne 0) { throw "Early fusion failed" }

# --- 2. LATE ---
Write-Host "=== [2/3] Running LATE fusion (GPU) ==="
& $python src/train.py `
  model.fusion_type=late
if ($LASTEXITCODE -ne 0) { throw "Late fusion failed" }

# --- 3. HYBRID ---
Write-Host "=== [3/3] Running HYBRID fusion (GPU) ==="
& $python src/train.py `
  model.fusion_type=hybrid
if ($LASTEXITCODE -ne 0) { throw "Hybrid fusion failed" }

# --- 4–6. HYBRID ATTENTION ABLATIONS ---
$heads = 1, 4, 8
$idx = 4
foreach ($h in $heads) {
    $expName = "a2_hybrid_heads_$h"
    Write-Host "=== [$idx/7] Training hybrid attention ablation: $h heads ==="
    & $python src/train.py `
        model.fusion_type=hybrid `
        model.num_heads=$h `
        experiment.name=$expName `
        experiment.save_dir=runs
    if ($LASTEXITCODE -ne 0) { throw "hybrid heads $h training failed" }
    $idx++
}

# --- 7. SINGLE MODALITY ---
Write-Host "=== [7/7] Training single modality from config/single_modality.yaml ==="
& $python src/train.py --config-name single_modality --config-dir config
if ($LASTEXITCODE -ne 0) { throw "single_modality training failed" }


Write-Host "=== All three runs finished! Check ./runs and ./experiments. ==="
