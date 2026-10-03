$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    throw "Environment not found. Run .\setup.ps1 first."
}
& ".\.venv\Scripts\python.exe" -m thirdeye @args
