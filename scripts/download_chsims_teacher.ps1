[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repo = 'Qwen/Qwen2.5-Omni-3B'
$revision = 'f61daeb0493a6e98cec1a58ddeec8f13c1121cc0'
$root = Split-Path -Parent $PSScriptRoot
$destination = Join-Path $root '.runtime/chsims-teacher/omni-3b'
New-Item -ItemType Directory -Force -Path $destination | Out-Null

$api = "https://www.modelscope.cn/api/v1/models/$repo/repo/files?Revision=$revision"
$release = Invoke-RestMethod -Uri $api -TimeoutSec 30
$shards = @{
    'model-00001-of-00003.safetensors' = @{ Size = 4994850880L; Sha256 = '349972cebff443030e9e96e1e930d7230979c6961cf046fa29fc580e4bcf49a6' }
    'model-00002-of-00003.safetensors' = @{ Size = 4999787448L; Sha256 = 'b29a76dfefb3aa33a5bd0faa19c681d039a44de57223653ea0e933df728c6d8c' }
}
$names = @('config.json', 'preprocessor_config.json', 'chat_template.json',
    'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'merges.txt',
    'vocab.json', 'added_tokens.json', 'generation_config.json', 'model.safetensors.index.json',
    'model-00001-of-00003.safetensors', 'model-00002-of-00003.safetensors')
$files = foreach ($name in $names) {
    if ($shards.ContainsKey($name)) {
        # ModelScope returns Git LFS pointers at the commit URL. Resolve the
        # binary from master and require the published digest for this release.
        [pscustomobject]@{ Name = $name; Size = $shards[$name].Size; Sha256 = $shards[$name].Sha256; Ref = 'master' }
        continue
    }
    $file = @($release.Data.Files | Where-Object { $_.Name -ceq $name -and $_.Sha256 -match '^[a-f0-9]{64}$' })
    if ($file.Count -ne 1) { throw "Missing published checksum for $name" }
    [pscustomobject]@{ Name = $name; Size = $file[0].Size; Sha256 = $file[0].Sha256; Ref = $revision }
}

$large = @($files | Where-Object Size -gt 100MB)
$parts = Join-Path $destination '.parts'
New-Item -ItemType Directory -Force -Path $parts | Out-Null
$rangeSize = 32MB
$jobs = @(foreach ($file in $large) {
    $path = Join-Path $destination $file.Name
    if ((Test-Path -LiteralPath $path) -and (Get-Item -LiteralPath $path).Length -eq $file.Size `
        -and (Get-FileHash -LiteralPath $path).Hash -eq $file.Sha256) { continue }
    for ($left = 0L; $left -lt $file.Size; $left += $rangeSize) {
        $right = [Math]::Min($file.Size, $left + $rangeSize) - 1
        [pscustomobject]@{ Name = $file.Name; Left = $left; Right = $right; Ref = $file.Ref }
    }
})
if ($jobs) { Write-Host "Fetching $(@($jobs).Count) checksum-bound weight ranges" }
$done = @($jobs | Where-Object { $null -ne $_ } | ForEach-Object -Parallel {
    $ErrorActionPreference = 'Stop'
    $part = Join-Path $using:parts "$($_.Name).$($_.Left)"
    $size = $_.Right - $_.Left + 1
    if (!(Test-Path -LiteralPath $part) -or (Get-Item -LiteralPath $part).Length -ne $size) {
        $url = "https://www.modelscope.cn/api/v1/models/$using:repo/repo?Revision=$($_.Ref)&FilePath=$($_.Name)"
        $http = & curl.exe --fail --silent --show-error --location --retry 2 --connect-timeout 15 --max-time 90 `
            --max-filesize $size --range "$($_.Left)-$($_.Right)" --output "$part.part" --write-out '%{http_code}' $url
        if ($LASTEXITCODE -ne 0 -or $http -ne '206' -or (Get-Item -LiteralPath "$part.part").Length -ne $size) {
            throw "Incomplete range for $($_.Name) at $($_.Left)"
        }
        Move-Item -LiteralPath "$part.part" -Destination $part -Force
    }
    $part
} -ThrottleLimit 8)
if ($done.Count -ne $jobs.Count) { throw 'Incomplete weight ranges. Re-run to resume.' }

foreach ($file in $large) {
    $path = Join-Path $destination $file.Name
    if ((Get-Item -LiteralPath $path -ErrorAction SilentlyContinue).Length -eq $file.Size `
        -and (Get-FileHash -LiteralPath $path).Hash -eq $file.Sha256) { continue }
    $assembled = "$path.assembled"
    $out = [IO.File]::Create($assembled)
    try {
        for ($left = 0L; $left -lt $file.Size; $left += $rangeSize) {
            $part = [IO.File]::OpenRead((Join-Path $parts "$($file.Name).$left"))
            try { $part.CopyTo($out) } finally { $part.Dispose() }
        }
    } finally { $out.Dispose() }
    if ((Get-FileHash -LiteralPath $assembled).Hash -ne $file.Sha256) { throw "SHA-256 did not match $assembled" }
    Move-Item -LiteralPath $assembled -Destination $path -Force
    Write-Host "Verified $($file.Name)"
}

$small = @($files | Where-Object Size -le 100MB)
$verified = @($small | ForEach-Object -Parallel {
    $ErrorActionPreference = 'Stop'
    $file = $_
    $path = Join-Path $using:destination $file.Name
    $part = "$path.part"
    if ((Test-Path -LiteralPath $path) -and (Get-FileHash -LiteralPath $path).Hash -eq $file.Sha256) { return $file.Name }
    if (!(Test-Path -LiteralPath $part) -or (Get-Item -LiteralPath $part).Length -lt $file.Size) {
        $url = "https://www.modelscope.cn/models/$using:repo/resolve/$($file.Ref)/$($file.Name)"
        Write-Host "Downloading $($file.Name) ($($file.Size) bytes)"
        & curl.exe --fail --silent --show-error --location --retry 3 --connect-timeout 20 `
            --continue-at - --output $part $url
        if ($LASTEXITCODE -ne 0) { throw "Download failed for $($file.Name), exit $LASTEXITCODE" }
    }
    if ((Get-Item -LiteralPath $part).Length -ne $file.Size -or (Get-FileHash -LiteralPath $part).Hash -ne $file.Sha256) {
        throw "Published SHA-256 or file size did not match $part"
    }
    Move-Item -LiteralPath $part -Destination $path -Force
    Write-Host "Verified $($file.Name)"
    $file.Name
} -ThrottleLimit 4)
if ($verified.Count -ne $small.Count) { throw 'The checkpoint metadata download is incomplete.' }

$receipt = [ordered]@{ model = $repo; revision = $revision; sha256 = [ordered]@{} }
foreach ($file in $files) { $receipt.sha256[$file.Name] = $file.Sha256 }
$receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $destination 'verified-model.json') -Encoding utf8

foreach ($file in $large) {
    for ($left = 0L; $left -lt $file.Size; $left += $rangeSize) {
        $part = Join-Path $parts "$($file.Name).$left"
        if (Test-Path -LiteralPath $part -PathType Leaf) { Remove-Item -LiteralPath $part }
    }
}
