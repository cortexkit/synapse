param(
    [Parameter(Mandatory)][string]$Archive,
    [switch]$RequireGpu
)
$ErrorActionPreference = 'Stop'
$root = Join-Path ([IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString())
$savedPath = $env:PATH
New-Item -ItemType Directory $root | Out-Null
function Invoke-Worker([string]$Exe, [string]$Argument) {
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $Exe
    $info.Arguments = $Argument
    $info.WorkingDirectory = Split-Path $Exe
    $info.UseShellExecute = $false
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $process = [Diagnostics.Process]::Start($info)
    $stdout = $process.StandardOutput.ReadToEndAsync()
    $stderr = $process.StandardError.ReadToEndAsync()
    try {
        if (!$process.WaitForExit(15000)) { $process.Kill(); throw 'Worker probe exceeded 15 seconds' }
        return @{ Code = $process.ExitCode; Out = $stdout.GetAwaiter().GetResult(); Err = $stderr.GetAwaiter().GetResult() }
    } finally { $process.Dispose() }
}
try {
    $package = Join-Path $root 'package'
    Expand-Archive $Archive $package
    $exe = Join-Path $package 'ck-synapse-worker-cuda.exe'
    $manifest = Get-Content (Join-Path $package 'manifest.json') -Raw | ConvertFrom-Json
    if ((Get-FileHash $exe -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifest.worker_sha256) { throw 'Worker hash mismatch' }
    foreach ($entry in $manifest.runtime_files) {
        if ((Get-FileHash (Join-Path $package $entry.file) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw "Hash mismatch: $($entry.file)" }
    }
    $empty = Join-Path $root 'no-sidecars'
    New-Item -ItemType Directory $empty | Out-Null
    Copy-Item $exe $empty
    $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
    $isolatedExe = Join-Path $empty 'ck-synapse-worker-cuda.exe'
    $version = Invoke-Worker $isolatedExe '--version'
    if ($version.Code -ne 0 -or $version.Out -notmatch '^ck-synapse-worker-cuda ') { throw "No-DLL version failed: $($version.Err)" }
    $missing = Invoke-Worker $isolatedExe '--probe-floor'
    $missingFloor = $missing.Out | ConvertFrom-Json
    if ($missing.Code -ne 2 -or $missingFloor.status -ne 'refused' -or $missingFloor.code -ne 'cuda_runtime_missing:cublasLt64_13.dll') { throw "Missing-DLL refusal failed: $($missing.Code) $($missing.Out) $($missing.Err)" }
    $present = Invoke-Worker $exe '--probe-floor'
    if ($present.Code -eq 0) {
        $floor = $present.Out | ConvertFrom-Json
        if ($floor.status -ne 'ok' -or $floor.code -ne 'ok' -or $floor.required.driver_api -ne 13020 -or $floor.observed.driver_api -lt 13020 -or $floor.observed.compute_capability.major -lt 7 -or ($floor.observed.compute_capability.major -eq 7 -and $floor.observed.compute_capability.minor -lt 5)) {
            throw 'GPU below owned-CUDA floor: driver API >= 13020 and compute capability >= 7.5 required'
        }
        Write-Output "PASS packaged GPU probe: $($present.Out.Trim())"
    } elseif ($RequireGpu) {
        throw "Packaged GPU probe failed: $($present.Code) $($present.Err)"
    } else {
        $floor = $present.Out | ConvertFrom-Json
        if ($present.Code -ne 2 -or $floor.status -ne 'refused' -or $floor.required.driver_api -ne 13020 -or $floor.code -notin @('cuda_no_driver', 'cuda_driver_too_old', 'cuda_compute_capability_too_low')) {
            throw "Packaged runtime resolution failed: $($present.Code) $($present.Out) $($present.Err)"
        }
        Write-Output "GPU execution not available on this runner: $($present.Out.Trim())"
    }
    Write-Output 'PASS archive hashes, no-DLL version, missing-DLL refusal, side-by-side runtime resolution'
} finally {
    $env:PATH = $savedPath
    Remove-Item $root -Recurse -Force
}
