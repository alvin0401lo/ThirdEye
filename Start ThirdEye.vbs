Set shell = CreateObject("WScript.Shell")
script = Replace(WScript.ScriptFullName, ".vbs", ".ps1")
shell.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & script & """", 0, False
