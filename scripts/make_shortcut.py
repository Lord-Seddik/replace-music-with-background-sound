r"""Create a "Music remover" shortcut that opens the window.

    .venv\Scripts\python.exe scripts\make_shortcut.py              (Desktop)
    .venv\Scripts\python.exe scripts\make_shortcut.py --out-dir FOLDER

Needs no PowerShell policy change: the one PowerShell call it makes uses
-ExecutionPolicy Bypass for that single process only.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

PS = (
    "$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:MR_LINK);"
    "$s.TargetPath=$env:MR_TARGET;$s.WorkingDirectory=$env:MR_ROOT;"
    "$s.IconLocation=$env:MR_TARGET+',0';"
    "$s.Description='Remove background music from videos';$s.Save()"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", help="folder for the shortcut (default: Desktop)")
    ap.add_argument("--name", default="Music remover")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root / ".venv" / "Scripts" / "music-remover-gui.exe"
    if not target.exists():
        print(f"Not found: {target}")
        print(r"Install first: .venv\Scripts\pip install -e .")
        return 1
    if a.out_dir:
        out = Path(a.out_dir).resolve()
    else:
        out = Path(subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
             "[Environment]::GetFolderPath('Desktop')"],
            capture_output=True, text=True, check=True).stdout.strip())
    out.mkdir(parents=True, exist_ok=True)
    link = out / f"{a.name}.lnk"
    env = dict(os.environ, MR_LINK=str(link), MR_TARGET=str(target), MR_ROOT=str(root))
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", PS],
                   env=env, check=True)
    print(f"Shortcut created: {link}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
