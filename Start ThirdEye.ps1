$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$LogDirectory = Join-Path $Root "logs"
$OutputLog = Join-Path $LogDirectory "thirdeye.log"
$ErrorLog = Join-Path $LogDirectory "thirdeye-error.log"

Add-Type -AssemblyName PresentationFramework
if (-not (Test-Path -LiteralPath $Python)) {
    [System.Windows.MessageBox]::Show("ThirdEye is not set up. Run setup.ps1 first.", "ThirdEye")
    exit 1
}

$CreatedNew = $false
$Mutex = [Threading.Mutex]::new($true, "Local\ThirdEyeAI", [ref]$CreatedNew)
if (-not $CreatedNew) {
    [System.Windows.MessageBox]::Show("ThirdEye is already running.", "ThirdEye")
    exit 0
}

New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$env:PYTHONUNBUFFERED = "1"
try {
    $Process = Start-Process `
        -FilePath $Python `
        -ArgumentList @("-m", "thirdeye", "--voice-control") `
        -WorkingDirectory $Root `
        -WindowStyle Hidden `
        -RedirectStandardOutput $OutputLog `
        -RedirectStandardError $ErrorLog `
        -Wait `
        -PassThru
    if ($Process.ExitCode -ne 0) {
        [System.Windows.MessageBox]::Show("ThirdEye stopped with an error. Check logs\thirdeye-error.log.", "ThirdEye")
    }
}
catch {
    $_ | Out-File $ErrorLog -Append
    [System.Windows.MessageBox]::Show("ThirdEye stopped with an error. Check logs\thirdeye-error.log.", "ThirdEye")
}
finally {
    $Mutex.ReleaseMutex()
    $Mutex.Dispose()
}
