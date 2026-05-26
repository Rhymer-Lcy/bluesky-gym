# Full retrain pipeline: 4 protagonists -> 4 adversaries -> 4 comparisons -> report.
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\run_full_retrain.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_full_retrain.ps1 -StartStage 2

param([int]$StartStage = 1)

$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

# Force UTF-8 stdout for child Python processes (avoids GBK on Windows cmd).
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# Python executable: set BSG_PYTHON env var, or copy .env.template -> .env.local
$envFile = Join-Path $PSScriptRoot ".." ".env.local"
if (Test-Path $envFile) {
    Get-Content $envFile | Where-Object { $_ -match '^BSG_PYTHON=' } | ForEach-Object {
        $env:BSG_PYTHON = ($_ -split '=', 2)[1].Trim()
    }
}
$py = if ($env:BSG_PYTHON) { $env:BSG_PYTHON } else { "python" }
$scenarios = @("head_on", "crossing", "merging", "overtaking")
$seeds = @{ head_on = 42; crossing = 43; merging = 44; overtaking = 45 }

function Stamp { Get-Date -Format "HH:mm:ss" }

function Find-LatestModel($pattern, $fileName) {
    $dir = Get-ChildItem "models" -Filter "${pattern}_*" -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | Select-Object -First 1
    if (-not $dir) { return $null }
    $path = Join-Path $dir.FullName $fileName
    if (Test-Path $path) { return $path } else { return $null }
}

function Abort($msg) { Write-Host "[$(Stamp)] ERROR: $msg"; exit 1 }

if ($StartStage -le 1) {
    Write-Host "[$(Stamp)] === STAGE 1/4: Protagonists (100k steps) ==="
    foreach ($s in $scenarios) {
        Write-Host "[$(Stamp)] --> protagonist [$s]"
        & $py scripts\train\train_adversarial.py --mode protagonist --scenario $s `
            --num-intruders 3 --timesteps 100000 --seed $seeds[$s] --device cpu
        if (-not (Find-LatestModel "protagonist_$s" "final_model.zip")) {
            Abort "protagonist $s final_model not produced"
        }
    }
}

if ($StartStage -le 2) {
    Write-Host "[$(Stamp)] === STAGE 2/4: Adversaries (50k steps) ==="
    foreach ($s in $scenarios) {
        $prot = Find-LatestModel "protagonist_$s" "best_model\best_model.zip"
        if (-not $prot) { $prot = Find-LatestModel "protagonist_$s" "final_model.zip" }
        if (-not $prot) { Abort "no protagonist model for $s" }
        Write-Host "[$(Stamp)] --> adversary [$s] vs $prot"
        & $py scripts\train\train_adversarial.py --mode adversary --scenario $s `
            --num-intruders 3 --timesteps 50000 --protagonist-model $prot `
            --seed $seeds[$s] --device cpu
        if (-not (Find-LatestModel "adversary_$s" "final_adversarial_policy.pt")) {
            Abort "adversary $s policy not produced"
        }
    }
}

if ($StartStage -le 3) {
    Write-Host "[$(Stamp)] === STAGE 3/4: Comparison (200 episodes) ==="
    foreach ($s in $scenarios) {
        $adv = Find-LatestModel "adversary_$s" "final_adversarial_policy.pt"
        if (-not $adv) { Abort "no adversary for $s" }
        Write-Host "[$(Stamp)] --> comparison [$s]"
        & $py scripts\eval\run_comparison_experiments.py --scenario $s `
            --num-intruders 3 --adversary-model $adv `
            --train-steps 10000 --eval-episodes 200 --seed $seeds[$s]
    }
}

Write-Host "[$(Stamp)] === STAGE 4/4: Report ==="
& $py scripts\report\generate_phase5_comprehensive_report.py

Write-Host "[$(Stamp)] === DONE ==="

# Usage tip: redirect output to logs/
# powershell -ExecutionPolicy Bypass -File scripts\run_full_retrain.ps1 *>&1 | Tee-Object logs\retrain_console.log
