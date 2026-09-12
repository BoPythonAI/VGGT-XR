$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$root='D:\VGGT-XR-artifacts'
$tempPath=Join-Path $root 'unity-temp'
New-Item -ItemType Directory -Path $tempPath -Force | Out-Null
$env:TEMP=$tempPath
$env:TMP=$tempPath
$env:UPM_CACHE_ROOT=Join-Path $root 'UnityCache'
$release=Invoke-RestMethod -Uri 'https://services.api.unity.com/unity/editor/release/v1/releases?version=2022.3.62f1&limit=1' -TimeoutSec 30
$download=$release.results[0].downloads | Where-Object { $_.platform -eq 'WINDOWS' -and $_.architecture -eq 'X86_64' } | Select-Object -First 1
$installer=Join-Path $root 'UnitySetup64-2022.3.62f1.exe'
$head=Invoke-WebRequest -Uri $download.url -Method Head -TimeoutSec 30
$expectedBytes=[long]($head.Headers['Content-Length'][0])
$target=Join-Path $root 'UnityEditor'
if((Get-PSDrive D).Free -lt 15GB){throw 'Need at least 15 GB free on D: for editor installation'}
if(-not(Test-Path -LiteralPath $installer) -or (Get-Item -LiteralPath $installer).Length -ne $expectedBytes) {
    Write-Output 'Downloading official Unity 2022.3 Editor to D:'
    Invoke-WebRequest -Uri $download.url -OutFile $installer -TimeoutSec 1800
}
if((Get-Item -LiteralPath $installer).Length -ne $expectedBytes){throw 'Unity installer size mismatch'}
$signature=Get-AuthenticodeSignature -LiteralPath $installer
if($signature.Status -ne 'Valid'){throw "Unity installer signature invalid: $($signature.Status)"}
Write-Output "Verified signature: $($signature.SignerCertificate.Subject)"
Write-Output 'Installing Unity Editor on D: using D: temporary files'
$process=Start-Process -FilePath $installer -ArgumentList @('/S',"/D=$target") -WindowStyle Hidden -Wait -PassThru
Write-Output "Installer exit code: $($process.ExitCode)"
if($process.ExitCode -ne 0){exit $process.ExitCode}
Get-Item -LiteralPath (Join-Path $target 'Editor\Unity.exe') | Select-Object FullName,Length
