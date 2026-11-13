param(
    [string]$PythonExe = "",
    [string]$IndexUrl = "https://pypi.tuna.tsinghua.edu.cn/simple",
    [switch]$SkipModels
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$env:PIP_CACHE_DIR = Join-Path $PSScriptRoot ".runtime\cache\pip"
$env:PYTHONPYCACHEPREFIX = Join-Path $PSScriptRoot ".runtime\cache\pycache"
$env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Path $env:PIP_CACHE_DIR -Force | Out-Null
if (!(Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    if ($PythonExe) {
        & $PythonExe -m venv .venv
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -m venv .venv
    } else {
        throw "Install Python 3.12 x64 first, or pass -PythonExe with its absolute path."
    }
    if ($LASTEXITCODE -ne 0) { throw "Could not create isolated Python 3.12 environment." }
}
$projectPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $projectPython -c "import sys; assert sys.version_info[:2] == (3,12), 'This project requires Python 3.12'; assert sys.prefix != sys.base_prefix, 'A project venv is required'"
if ($LASTEXITCODE -ne 0) { throw "Invalid Python environment. No system packages were changed." }
if (Test-Path -LiteralPath "requirements-lock.txt") {
    & $projectPython -m pip install --disable-pip-version-check --index-url $IndexUrl -r requirements-lock.txt
} else {
    & $projectPython -m pip install --disable-pip-version-check --index-url $IndexUrl -r requirements-base.txt -r requirements-models.txt
}
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. Rerun this script to resume." }
& $projectPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency conflict detected." }
if (!$SkipModels) {
    & $projectPython scripts/download_models.py --all
    if ($LASTEXITCODE -ne 0) { throw "Model download incomplete. Rerun this script to resume; simulation remains usable." }
}
& $projectPython scripts/doctor.py
Write-Host "Ready. Double-click start.cmd to open the local desktop app."
