param(
    [string]$DataRoot = "data\owl_output",
    [string]$LearningRoot = "data\owl_output\learning",
    [string]$PythonExe = "python",
    [ValidateSet("dino", "clip")]
    [string]$EmbeddingBackend = "dino",
    [string]$ClipModelId = "openai/clip-vit-base-patch32",
    [string]$DinoModelId = "facebook/dinov2-base"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    Write-Host "$PythonExe $($Arguments -join ' ')" -ForegroundColor DarkGray

    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name"
    }
}

$embeddingsDir = Join-Path $LearningRoot "embeddings"
$embeddingScript = if ($EmbeddingBackend -eq "dino") { "scripts/extract_dino_embeddings.py" } else { "scripts/extract_clip_embeddings.py" }
$embeddingArchive = Join-Path $embeddingsDir ("{0}_embeddings_all.npz" -f $EmbeddingBackend)
$embeddingModelId = if ($EmbeddingBackend -eq "dino") { $DinoModelId } else { $ClipModelId }

Invoke-Step -Name "Rebuild staged dataset" -Arguments @(
    "data_pipeline.py",
    "--data-root", $DataRoot,
    "--stages", "all",
    "--overwrite"
)

Invoke-Step -Name "Build learning manifests" -Arguments @(
    "scripts/build_manifest.py",
    "--data-root", $DataRoot,
    "--output-dir", $LearningRoot
)

Invoke-Step -Name ("Extract {0} embeddings" -f $EmbeddingBackend.ToUpperInvariant()) -Arguments @(
    $embeddingScript,
    "--manifest-frames", (Join-Path $LearningRoot "manifest_frames.csv"),
    "--output-dir", $embeddingsDir,
    "--model-id", $embeddingModelId
)

Invoke-Step -Name "Run embedding diagnostics" -Arguments @(
    "scripts/embedding_diagnostics.py",
    "--embeddings-npz", $embeddingArchive,
    "--output-dir", (Join-Path $LearningRoot "diagnostics")
)

Invoke-Step -Name "Train latent transition baseline" -Arguments @(
    "scripts/train_transition_baseline.py",
    "--embeddings-npz", $embeddingArchive,
    "--transitions-csv", (Join-Path $LearningRoot "manifest_transitions.csv"),
    "--output-dir", (Join-Path $LearningRoot "ar_baseline")
)

Write-Host ""
Write-Host "Core pipeline rebuild complete." -ForegroundColor Green
