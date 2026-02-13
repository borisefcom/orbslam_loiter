param(
    [Parameter(Mandatory = $false)]
    [string]$TargetHost = "10.92.44.50",

    [Parameter(Mandatory = $false)]
    [string]$TargetUser = "esp710",

    [Parameter(Mandatory = $false)]
    [string]$TargetBaseDir = "~/vo_loiter",

    [Parameter(Mandatory = $false)]
    [string]$JumpHost = "",

    [Parameter(Mandatory = $false)]
    [string]$JumpUser = "",

    [Parameter(Mandatory = $false)]
    [switch]$NoJump
)

$ErrorActionPreference = "Stop"

function ConvertTo-PlainText([Security.SecureString]$Secure) {
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Secure)
    try { return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
}

$useJump = (-not $NoJump) -and ($JumpHost.Trim().Length -gt 0) -and ($JumpUser.Trim().Length -gt 0)

$remoteBaseDir = $TargetBaseDir.Trim()
if ($remoteBaseDir -eq "~") {
    $remoteBaseDir = ("/home/{0}" -f $TargetUser)
} elseif ($remoteBaseDir.StartsWith("~/")) {
    $remoteBaseDir = ("/home/{0}/{1}" -f $TargetUser, $remoteBaseDir.Substring(2))
}

Write-Host ("Target: {0}@{1} ({2})" -f $TargetUser, $TargetHost, $remoteBaseDir)
if ($useJump) {
    Write-Host ("Jump:   {0}@{1}" -f $JumpUser, $JumpHost)
} else {
    Write-Host "Jump:   (none)"
}

$targetPw = ConvertTo-PlainText (Read-Host -AsSecureString ("SSH password for {0}@{1}" -f $TargetUser, $TargetHost))
$jumpPw = ""
if ($useJump) {
    $jumpPw = ConvertTo-PlainText (Read-Host -AsSecureString ("SSH password for {0}@{1}" -f $JumpUser, $JumpHost))
}

$proxyArg = @()
if ($useJump) {
    $proxyCmd = ("plink -ssh -batch -pw {0} {1}@{2} -nc %host:%port" -f $jumpPw, $JumpUser, $JumpHost)
    $proxyArg = @("-proxycmd", $proxyCmd)
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$remoteTar = ("/home/{0}/indoor_loiter_logs_{1}.tgz" -f $TargetUser, $stamp)

$localOutDir = Join-Path $PSScriptRoot ("_rpi_logs\\{0}" -f $stamp)
New-Item -ItemType Directory -Force -Path $localOutDir | Out-Null
$localTar = Join-Path $localOutDir ("indoor_loiter_logs_{0}.tgz" -f $stamp)

$remoteCmd = @"
set -e
cd "$remoteBaseDir/indoor_loiter"
mkdir -p logs
tar -czf "$remoteTar" logs
"@

Write-Host "Packing logs on target..."
& plink -batch @proxyArg -pw $targetPw ("{0}@{1}" -f $TargetUser, $TargetHost) $remoteCmd
if ($LASTEXITCODE -ne 0) {
    throw ("plink remote tar failed with exit code {0}" -f $LASTEXITCODE)
}

Write-Host ("Downloading {0} -> {1}" -f $remoteTar, $localTar)
& pscp -batch @proxyArg -pw $targetPw ("{0}@{1}:{2}" -f $TargetUser, $TargetHost, $remoteTar) $localTar
if ($LASTEXITCODE -ne 0) {
    throw ("pscp download failed with exit code {0}" -f $LASTEXITCODE)
}

Write-Host "Cleaning up remote..."
& plink -batch @proxyArg -pw $targetPw ("{0}@{1}" -f $TargetUser, $TargetHost) ("rm -f ""{0}""" -f $remoteTar)
if ($LASTEXITCODE -ne 0) {
    throw ("plink remote cleanup failed with exit code {0}" -f $LASTEXITCODE)
}

Write-Host "Extracting..."
Push-Location $localOutDir
try {
    & tar -xzf $localTar
} finally {
    Pop-Location
}

Write-Host ("Done. Local logs: {0}" -f $localOutDir)
