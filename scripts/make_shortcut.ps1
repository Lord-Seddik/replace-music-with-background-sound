<#
Creates a shortcut to the music-remover window (music-remover-gui.exe).
By default it goes on your Desktop; use -OutDir to put it elsewhere.

  powershell -ExecutionPolicy Bypass -File scripts\make_shortcut.ps1
  powershell -ExecutionPolicy Bypass -File scripts\make_shortcut.ps1 -OutDir C:\Some\Folder
#>
param(
    [string]$OutDir = [Environment]::GetFolderPath("Desktop"),
    [string]$Name = "Music remover"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$target = Join-Path $root ".venv\Scripts\music-remover-gui.exe"
if (-not (Test-Path $target)) {
    throw "Not found: $target. Install the project first: .venv\Scripts\pip install -e ."
}
New-Item -ItemType Directory -Force $OutDir | Out-Null
$link = Join-Path $OutDir "$Name.lnk"
$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($link)
$sc.TargetPath = $target
$sc.WorkingDirectory = $root
$sc.Description = "Remove background music from videos"
$sc.Save()
Write-Output $link
