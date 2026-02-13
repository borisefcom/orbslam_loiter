param(
    [Parameter(Mandatory = $false)]
    [string]$TargetHost = "10.92.44.50",

    [Parameter(Mandatory = $false)]
    [string]$TargetUser = "esp710",

    # Remote base dir that should contain `indoor_loiter/` and `orbslam/`.
    [Parameter(Mandatory = $false)]
    [string]$TargetBaseDir = "~/vo_loiter",

    # Optional SSH hop (ProxyCommand via PuTTY/plink).
    # Example: JumpUser="xtend_m2", JumpHost="100.86.78.11"
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

# Expand "~" on the remote side, because tilde expansion does not occur when quoted.
$remoteBaseDir = $TargetBaseDir.Trim()
if ($remoteBaseDir -eq "~") {
    $remoteBaseDir = ("/home/{0}" -f $TargetUser)
} elseif ($remoteBaseDir.StartsWith("~/")) {
    $remoteBaseDir = ("/home/{0}/{1}" -f $TargetUser, $remoteBaseDir.Substring(2))
}

Write-Host ("Target: {0}@{1} -> {2}" -f $TargetUser, $TargetHost, $remoteBaseDir)
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

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$tarPath = Join-Path $env:TEMP ("vo_loiter_payload_{0}.tgz" -f $stamp)

Write-Host ("Creating payload: {0}" -f $tarPath)
Push-Location $repoRoot
    try {
        $excludes = @(
        "--exclude=indoor_loiter/indoor-osd-app",
        "--exclude=indoor_loiter/_rpi_logs",
        "--exclude=indoor_loiter/__pycache__",
        "--exclude=indoor_loiter/logs",
        "--exclude=orbslam/.tmp",
        "--exclude=orbslam/logs",
        "--exclude=orbslam/__pycache__",
        "--exclude=orbslam/.venv",
        "--exclude=orbslam/.idea",
        "--exclude=orbslam/third_party/ORB_SLAM3_pybind/python_wrapper/*.dll",
        "--exclude=orbslam/third_party/ORB_SLAM3_pybind/python_wrapper/*.pyd",
        "--exclude=orbslam/third_party/ORB_SLAM3_pybind/python_wrapper/*.lib",
        "--exclude=orbslam/third_party/ORB_SLAM3_pybind/python_wrapper/*.exp"
    )
    & tar @excludes -czf $tarPath "indoor_loiter" "orbslam"
    if ($LASTEXITCODE -ne 0) {
        throw ("tar failed with exit code {0}" -f $LASTEXITCODE)
    }
    if (-not (Test-Path -LiteralPath $tarPath)) {
        throw ("tar did not create expected payload: {0}" -f $tarPath)
    }
} finally {
    Pop-Location
}

$remoteTar = ("/home/{0}/vo_loiter_payload.tgz" -f $TargetUser)

$proxyArg = @()
if ($useJump) {
    $proxyCmd = ("plink -ssh -batch -pw {0} {1}@{2} -nc %host:%port" -f $jumpPw, $JumpUser, $JumpHost)
    $proxyArg = @("-proxycmd", $proxyCmd)
}

Write-Host ("Uploading to {0}" -f $remoteTar)
& pscp -batch @proxyArg -pw $targetPw $tarPath ("{0}@{1}:{2}" -f $TargetUser, $TargetHost, $remoteTar)
if ($LASTEXITCODE -ne 0) {
    throw ("pscp upload failed with exit code {0}" -f $LASTEXITCODE)
}

$remoteCmd = @"
set -e
mkdir -p "$remoteBaseDir"
tar -xzf "$remoteTar" -C "$remoteBaseDir"
rm -f "$remoteTar"
chmod +x "$remoteBaseDir/indoor_loiter/setup_pi5.sh" 2>/dev/null || true
"@

Write-Host "Extracting on target..."
& plink -batch @proxyArg -pw $targetPw ("{0}@{1}" -f $TargetUser, $TargetHost) $remoteCmd
if ($LASTEXITCODE -ne 0) {
    throw ("plink remote extract failed with exit code {0}" -f $LASTEXITCODE)
}

Write-Host "Done. Next (on target):"
Write-Host ("  cd {0}/indoor_loiter && python3 server.py" -f $remoteBaseDir)
