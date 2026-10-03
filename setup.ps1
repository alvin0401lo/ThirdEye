$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

$Python = $null
$SystemPython = Get-Command python -ErrorAction SilentlyContinue
if ($SystemPython) {
    & $SystemPython.Source --version *> $null
    if ($LASTEXITCODE -eq 0) {
        $Python = $SystemPython.Source
    }
}
if (-not $Python -and (Test-Path -LiteralPath $BundledPython)) {
    $Python = $BundledPython
}
if (-not $Python) {
    throw "Python 3.11-3.13 is required. Install Python, then run setup.ps1 again."
}

Set-Location $ProjectRoot
$VenvPython = ".venv\Scripts\python.exe"
$CreateVenv = -not (Test-Path -LiteralPath $VenvPython)
if (-not $CreateVenv) {
    $VenvConfig = ".venv\pyvenv.cfg"
    $HomeLine = Get-Content -LiteralPath $VenvConfig -ErrorAction SilentlyContinue |
        Where-Object { $_ -match "^home\s*=\s*(.+)$" } |
        Select-Object -First 1
    if (-not $HomeLine) {
        $CreateVenv = $true
    } else {
        $BasePython = Join-Path (($HomeLine -split "=", 2)[1].Trim()) "python.exe"
        $CreateVenv = -not (Test-Path -LiteralPath $BasePython)
    }
}
if ($CreateVenv) {
    if (Test-Path -LiteralPath ".venv") {
        $BackupName = ".venv-broken-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
        Move-Item -LiteralPath ".venv" -Destination $BackupName
        Write-Host "Broken virtual environment moved to $BackupName"
    }
    & $Python -m venv .venv
}
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -e "."

$ModelDirectory = Join-Path $ProjectRoot "models"
$HandModel = Join-Path $ModelDirectory "hand_landmarker.task"
if (-not (Test-Path -LiteralPath $HandModel)) {
    New-Item -ItemType Directory -Path $ModelDirectory -Force | Out-Null
    Invoke-WebRequest `
        -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" `
        -OutFile $HandModel
}
Write-Host "Setup complete. Run .\run.ps1"
