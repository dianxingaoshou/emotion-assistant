param([switch]$PullModel, [switch]$PullEmbedding)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$portableExe = Join-Path $projectDir ".runtime/ollama/ollama.exe"
$usingPortable = Test-Path -LiteralPath $portableExe -PathType Leaf
$api = "http://127.0.0.1:11434"

function Get-LocalModels {
    try {
        return Invoke-RestMethod -Uri "$api/api/tags" -TimeoutSec 3
    } catch {
        return $null
    }
}

$installed = Get-LocalModels
$previous = @{}
foreach ($key in "OLLAMA_HOST", "OLLAMA_MODELS", "OLLAMA_NO_CLOUD") {
    $previous[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
}
try {
    if ($null -eq $installed -or $PullModel -or $PullEmbedding) {
        if ($usingPortable) {
            $ollamaExe = $portableExe
        } else {
            $ollamaExe = (Get-Command ollama.exe -CommandType Application -ErrorAction Stop).Source
        }
        $env:OLLAMA_HOST = "127.0.0.1:11434"
    }
    if ($null -eq $installed) {
        if ($usingPortable) {
            $env:OLLAMA_MODELS = Join-Path $projectDir "models/ollama"
        }
        $env:OLLAMA_NO_CLOUD = "1"
        $server = Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden -PassThru
        for ($attempt = 0; $attempt -lt 30; $attempt++) {
            Start-Sleep -Milliseconds 500
            $installed = Get-LocalModels
            if ($null -ne $installed -or $server.HasExited) { break }
        }
        if ($null -eq $installed) { throw "The local Ollama service did not start on $api." }
    }
    if ((-not $PullEmbedding -or $PullModel) -and @($installed.models.name) -notcontains "qwen3:8b") {
        if (-not $PullModel) { throw "Local service is ready. Run this script with -PullModel to download qwen3:8b." }
        & $ollamaExe pull qwen3:8b
        if ($LASTEXITCODE -ne 0) { throw "Download of qwen3:8b failed." }
    }
    if ($PullEmbedding) {
        $embedding = @($installed.models | Where-Object { $_.name -eq "qwen3-embedding:0.6b" })
        if ($embedding.Count -eq 0) {
            & $ollamaExe pull qwen3-embedding:0.6b
            if ($LASTEXITCODE -ne 0) {
                throw "Download failed. For the project portable server, run .\.venv\Scripts\python.exe scripts/install_foundation_embedding.py and retry."
            }
            $embedding = @((Get-LocalModels).models | Where-Object { $_.name -eq "qwen3-embedding:0.6b" })
        }
        if ($embedding.Count -ne 1 -or $embedding[0].digest -ne "ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d") {
            throw "The installed embedding model differs from the trained sentiment head."
        }
        Write-Output "Local sentiment encoder ready: qwen3-embedding:0.6b at $api"
    }
} finally {
    foreach ($key in $previous.Keys) {
        [Environment]::SetEnvironmentVariable($key, $previous[$key], "Process")
    }
}
if (-not $PullEmbedding -or $PullModel) { Write-Output "Local text review is ready: qwen3:8b at $api" }
