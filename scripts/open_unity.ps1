param(
    [string]$UnityEditor='D:\VGGT-XR-artifacts\UnityEditor\Editor\Unity.exe',
    [string]$ProjectPath='D:\VGGT-XR-artifacts\UnityProject'
)
$ErrorActionPreference='Stop'
if(-not(Test-Path -LiteralPath $UnityEditor)){throw 'Install and activate Unity Editor first. The prepared project remains on D:.'}
$taskCache=Join-Path $ProjectPath '..\UnityCache'
$taskTemp=Join-Path $ProjectPath '..\unity-temp'
New-Item -ItemType Directory -Path $taskCache,$taskTemp -Force | Out-Null
$env:UPM_CACHE_ROOT=[System.IO.Path]::GetFullPath($taskCache)
$env:TEMP=[System.IO.Path]::GetFullPath($taskTemp)
$env:TMP=$env:TEMP
Start-Process -FilePath $UnityEditor -ArgumentList @('-force-d3d12','-projectPath',('"'+$ProjectPath+'"')) -WindowStyle Normal
