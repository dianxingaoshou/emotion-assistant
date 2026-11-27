param([switch]$PullModel)

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
    if ($null -eq $installed -or $PullModel) {
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
    if (@($installed.models.name) -notcontains "qwen3:8b") {
        if (-not $PullModel) { throw "Local service is ready. Run this script with -PullModel to download qwen3:8b." }
        & $ollamaExe pull qwen3:8b
        if ($LASTEXITCODE -ne 0) { throw "Download of qwen3:8b failed." }
    }
} finally {
    foreach ($key in $previous.Keys) {
        [Environment]::SetEnvironmentVariable($key, $previous[$key], "Process")
    }
}
Write-Output "Local text review is ready: qwen3:8b at $api"
