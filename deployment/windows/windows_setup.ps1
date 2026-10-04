$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $ProjectRoot

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python Launcher was not found. Install 64-bit Python 3.12, then rerun this script."
}
py -3.12 -c "import sys; assert (3, 11) <= sys.version_info[:2] < (3, 14), sys.version; print(sys.version)"
if ($LASTEXITCODE -ne 0) { throw "Python 3.12 is required." }

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create the Python virtual environment." }
}
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "Could not update pip tooling." }
& $Python -m pip install -e .
if ($LASTEXITCODE -ne 0) { throw "Could not install ThirdEye dependencies." }

New-Item -ItemType Directory -Force -Path "models" | Out-Null
$HandModel = Join-Path $ProjectRoot "models\hand_landmarker.task"
if (-not (Test-Path $HandModel)) {
    $TemporaryModel = "$HandModel.part"
    try {
        Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile $TemporaryModel
        Move-Item -Force $TemporaryModel $HandModel
    }
    finally {
        if (Test-Path $TemporaryModel) { Remove-Item $TemporaryModel }
    }
}

if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Copy-Item ".env.example" ".env"
}
Write-Host "Setup complete. Edit .env, then run .\deployment\windows\windows_run.ps1 --device-test"
