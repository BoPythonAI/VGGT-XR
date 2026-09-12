param(
    [Parameter(Mandatory=$true)][string]$Artifacts,
    [string]$ProjectPath = 'D:\VGGT-XR-artifacts\UnityProject',
    [string]$UnityEditor,
    [string]$OfflineGaussianPackage,
    [switch]$Build
)
$ErrorActionPreference='Stop'
$source=Join-Path $PSScriptRoot '..\unity\VGGT_XR'
if ([System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($ProjectPath)) -eq 'C:\') { throw 'Put the Unity project and its Library on D: or another data drive.' }
New-Item -ItemType Directory -Path $ProjectPath -Force | Out-Null
Copy-Item -Path (Join-Path $source '*') -Destination $ProjectPath -Recurse -Force
if ($OfflineGaussianPackage) {
    $package=Join-Path $ProjectPath 'Packages\org.nesnausk.gaussian-splatting'
    New-Item -ItemType Directory -Path $package -Force | Out-Null
    Copy-Item -Path (Join-Path $OfflineGaussianPackage '*') -Destination $package -Recurse -Force
}
$streaming=Join-Path $ProjectPath 'Assets\StreamingAssets'
New-Item -ItemType Directory -Path $streaming -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $Artifacts 'gaussian\scene.ply') -Destination (Join-Path $streaming 'scene.ply') -Force
Copy-Item -LiteralPath (Join-Path $Artifacts 'gaussian\unity_camera.json') -Destination $streaming -Force
Copy-Item -LiteralPath (Join-Path $Artifacts 'geometry\unity_points.ply') -Destination (Join-Path $streaming 'points.ply') -Force
Copy-Item -LiteralPath (Join-Path $Artifacts 'geometry\confidence.ply') -Destination $streaming -Force
if ($UnityEditor) {
    $env:UPM_CACHE_ROOT=Join-Path $ProjectPath '..\UnityCache'
    $method=if($Build){'BuildDemo.BuildWindows'}else{'BuildDemo.CreateScene'}
    & $UnityEditor -batchmode -quit -force-d3d12 -projectPath $ProjectPath -executeMethod $method -logFile (Join-Path $ProjectPath 'build.log')
    if($LASTEXITCODE -ne 0){throw "Unity failed. See $ProjectPath\build.log"}
} else {
    Write-Output "Project ready: $ProjectPath. Open with activated Unity 2022.3, then VGGT-XR > Create Desktop Demo Scene."
}
