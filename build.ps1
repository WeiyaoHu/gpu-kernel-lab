param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Source,

    [string]$Architecture = "sm_89"
)

$ErrorActionPreference = "Stop"

$sourcePath = if ([System.IO.Path]::IsPathRooted($Source)) {
    [System.IO.Path]::GetFullPath($Source)
}
else {
    [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $Source))
}

if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
    throw "CUDA source file not found: $sourcePath"
}

if ([System.IO.Path]::GetExtension($sourcePath) -ne ".cu") {
    throw "Expected a .cu source file: $sourcePath"
}

$cudaPath = [Environment]::GetEnvironmentVariable("CUDA_PATH", "Machine")
if (-not $cudaPath) {
    throw "CUDA_PATH is not configured. Install the NVIDIA CUDA Toolkit first."
}

$nvccPath = Join-Path $cudaPath "bin\nvcc.exe"
if (-not (Test-Path -LiteralPath $nvccPath -PathType Leaf)) {
    throw "nvcc.exe not found: $nvccPath"
}

$vswherePath = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vswherePath -PathType Leaf)) {
    throw "vswhere.exe not found. Install Visual Studio 2022 Build Tools with the C++ workload."
}

$visualStudioPath = & $vswherePath `
    -latest `
    -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath

if (-not $visualStudioPath) {
    throw "Visual Studio C++ Build Tools were not found."
}

$vcvarsPath = Join-Path $visualStudioPath "VC\Auxiliary\Build\vcvars64.bat"
$buildPath = Join-Path $PSScriptRoot "build"
$outputPath = Join-Path $buildPath (([System.IO.Path]::GetFileNameWithoutExtension($sourcePath)) + ".exe")

New-Item -ItemType Directory -Force -Path $buildPath | Out-Null

$compileCommand = 'call "{0}" >nul && "{1}" -std=c++17 -O2 -arch={2} -Xcompiler=/utf-8 "{3}" -o "{4}"' -f `
    $vcvarsPath, $nvccPath, $Architecture, $sourcePath, $outputPath

& "$env:SystemRoot\System32\cmd.exe" /d /s /c $compileCommand
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Built: $outputPath"
