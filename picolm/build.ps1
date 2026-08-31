Set-Location "D:\picolm\picolm"
$proc = Start-Process "C:\Windows\System32\cmd.exe" -ArgumentList "/c D:\picolm\picolm\.hunger_build.bat" -RedirectStandardOutput "build_ps.log" -RedirectStandardError "build_ps_err.log" -Wait -PassThru
Write-Output "ExitCode: $($proc.ExitCode)"
