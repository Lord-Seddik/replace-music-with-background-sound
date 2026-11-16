"""A small drop-files window around the `music-remover` command.

The window runs the command as a subprocess and reads its status line, so the
pipeline code is not involved. The parts that do not need a screen (building
the argument list, parsing progress, running and cancelling the subprocess)
are plain functions and classes at the top; the tkinter window is below.
"""
from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .mix import parse_range

# the same set the command uses (cli.py is not imported: it pulls in torch)
VIDEO_EXTENSIONS = {".mkv", ".mp4", ".m4v", ".mov", ".webm"}
FILL_LEVEL_LIMIT_DB = 20
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- pure parts

@dataclass
class Options:
    files: list[Path] = field(default_factory=list)
    out_dir: Path | None = None
    fill: bool = False
    library: Path | None = None
    fill_level: float = 0.0
    keep: str = ""
    keep_original: bool = False
    device: str | None = None  # None: the command's own default (auto)


def split_keep(text: str) -> list[str]:
    """'0:00-1:30, 22:10-end' -> ['0:00-1:30', '22:10-end'] (commas, semicolons or
    new lines separate the ranges). Raises ValueError naming the bad range."""
    ranges = [part.strip() for part in re.split(r"[,;\n]", text) if part.strip()]
    for r in ranges:
        try:
            parse_range(r)
        except ValueError as exc:
            raise ValueError(f"keep range '{r}': {exc}") from None
    return ranges


def build_args(opts: Options) -> list[str]:
    """The arguments after `music-remover`; a list, so paths with spaces need no quoting."""
    if not opts.files:
        raise ValueError("add at least one video file or folder")
    if opts.fill and opts.library is None:
        raise ValueError("choose the ambience library folder, or turn the fill off")
    args = [str(p) for p in opts.files]
    if opts.out_dir is not None:
        args += ["-o", str(opts.out_dir)]
    for r in split_keep(opts.keep):
        args += ["--keep", r]
    if opts.keep_original:
        args.append("--keep-original")
    if opts.fill:
        args += ["--fill", "library", "--library", str(opts.library)]
        level = max(-FILL_LEVEL_LIMIT_DB, min(FILL_LEVEL_LIMIT_DB, opts.fill_level))
        if level != 0:
            args += ["--fill-level", f"{level:g}"]
    if opts.device:
        args += ["--device", opts.device]
    return args


def python_exe() -> str:
    """python.exe, not pythonw.exe (the GUI script runs under pythonw; a child of that
    would have no stdout)."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and (exe.with_name("python.exe")).is_file():
        return str(exe.with_name("python.exe"))
    return str(exe)


def build_command(opts: Options) -> list[str]:
    return [python_exe(), "-m", "musicremover", *build_args(opts)]


def show_command(cmd: list[str]) -> str:
    """The command as one line that could be pasted into a terminal."""
    return subprocess.list2cmdline(cmd)


def expand_inputs(paths: list[Path]) -> list[Path]:
    """Folders become the videos in them (same rule as the command); other paths are kept."""
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out += sorted((c for c in p.iterdir()
                           if c.is_file() and c.suffix.lower() in VIDEO_EXTENSIONS
                           and ".nomusic." not in c.name), key=lambda c: c.name)
        else:
            out.append(p)
    return out


@dataclass
class Progress:
    index: int
    total: int
    name: str
    step: str
    percent: int | None = None
    eta: str | None = None


_PREFIX = re.compile(r"^\[(\d+)/(\d+)\] (.*?)  (.*\S)\s*$")
_PERCENT = re.compile(r"^(.*?) (\d+)%(?:  eta (\S+))?$")


def parse_progress(line: str) -> Progress | None:
    """'[1/2] name.mkv  separating 12%  eta 5:20' -> Progress(1, 2, 'name.mkv', 'separating', 12, '5:20').
    Lines of other kinds (results, errors) give None."""
    m = _PREFIX.match(line.strip())
    if not m:
        return None
    rest = m.group(4)
    p = _PERCENT.match(rest)
    if p:
        return Progress(int(m.group(1)), int(m.group(2)), m.group(3), p.group(1), int(p.group(2)), p.group(3))
    return Progress(int(m.group(1)), int(m.group(2)), m.group(3), rest)


def split_lines(buffer: bytes) -> tuple[list[str], bytes]:
    """Split on \\r and \\n; returns (complete lines, unfinished tail)."""
    parts = re.split(rb"[\r\n]+", buffer)
    tail = parts.pop()
    return [p.decode("utf-8", "replace") for p in parts if p.strip()], tail


# ------------------------------------------------------------- job runner

class JobRunner:
    """Runs the command on a thread; the owner polls `events` (a queue) from its own thread.

    Events: ("progress", Progress), ("line", str), ("done", returncode, cancelled)."""

    def __init__(self, cmd: list[str], cwd: Path | None = None) -> None:
        self.cmd = cmd
        self.cwd = cwd
        self.events: queue.Queue = queue.Queue()
        self.proc: subprocess.Popen | None = None
        self.cancelled = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, cwd=self.cwd, env=env,
                                         creationflags=flags)
        except OSError as exc:
            self.events.put(("line", f"could not start the command: {exc}"))
            self.events.put(("done", -1, False))
            return
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        proc = self.proc
        assert proc is not None and proc.stdout is not None
        buf = b""
        while True:
            chunk = os.read(proc.stdout.fileno(), 4096)
            if not chunk:
                break
            lines, buf = split_lines(buf + chunk)
            for ln in lines:
                self._emit(ln)
        for ln in split_lines(buf + b"\n")[0]:
            self._emit(ln)
        self.events.put(("done", proc.wait(), self.cancelled))

    def _emit(self, line: str) -> None:
        prog = parse_progress(line)
        self.events.put(("progress", prog) if prog else ("line", line))

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def cancel(self) -> None:
        """Stop the command and anything it started (ffmpeg)."""
        if not self.running():
            return
        self.cancelled = True
        assert self.proc is not None
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                           capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            self.proc.terminate()

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)


# ------------------------------------------------------------------ window

def _open_folder(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 - opens the folder in Explorer
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)])
    else:
        subprocess.run(["xdg-open", str(path)])


def main() -> int:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
        root = TkinterDnD.Tk()
        can_drop = True
    except Exception:  # missing package or tkdnd library: the buttons still work
        root = tk.Tk()
        DND_FILES = None
        can_drop = False
    root.title("Music remover")
    root.minsize(640, 600)
    pad = {"padx": 8, "pady": 4}

    rows: dict[str, Path] = {}     # tree item id -> file
    state: dict = {"runner": None, "out_dir": None, "errors": [], "tail": []}

    # --- files
    top = ttk.LabelFrame(root, text="Videos" + ("  (drop files or a folder here)" if can_drop else ""))
    top.pack(fill="both", expand=True, **pad)
    tree = ttk.Treeview(top, columns=("status",), height=7)
    tree.heading("#0", text="File")
    tree.heading("status", text="Progress")
    tree.column("#0", width=330)
    tree.column("status", width=250)
    tree.pack(side="left", fill="both", expand=True, padx=4, pady=4)
    btns = ttk.Frame(top)
    btns.pack(side="right", fill="y", padx=4, pady=4)

    def set_status(item: str, text: str) -> None:
        tree.set(item, "status", text)

    def add_paths(paths: list[Path]) -> None:
        if state["runner"] is not None:
            return
        known = set(rows.values())
        for p in expand_inputs(paths):
            if p not in known and p.suffix.lower() in VIDEO_EXTENSIONS:
                item = tree.insert("", "end", text=p.name)
                rows[item] = p
                set_status(item, "waiting")
                known.add(p)

    def add_files() -> None:
        names = filedialog.askopenfilenames(title="Choose videos",
                                            filetypes=[("Video", "*.mkv *.mp4 *.m4v *.mov *.webm"), ("All", "*.*")])
        add_paths([Path(n) for n in names])

    def add_folder() -> None:
        d = filedialog.askdirectory(title="Choose a folder of videos")
        if d:
            add_paths([Path(d)])

    def remove_selected() -> None:
        if state["runner"] is None:
            for item in tree.selection():
                rows.pop(item, None)
                tree.delete(item)

    def clear_all() -> None:
        if state["runner"] is None:
            for item in list(rows):
                tree.delete(item)
            rows.clear()

    for text, cmd in (("Add files...", add_files), ("Add folder...", add_folder),
                      ("Remove", remove_selected), ("Clear", clear_all)):
        ttk.Button(btns, text=text, command=cmd).pack(fill="x", pady=2)

    if can_drop:
        def on_drop(event) -> None:
            add_paths([Path(s) for s in root.tk.splitlist(event.data)])
        for w in (root, tree, top):
            w.drop_target_register(DND_FILES)
            w.dnd_bind("<<Drop>>", on_drop)

    # --- options
    opt = ttk.LabelFrame(root, text="Options")
    opt.pack(fill="x", **pad)
    opt.columnconfigure(1, weight=1)
    out_var, lib_var, keep_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
    fill_var, orig_var = tk.BooleanVar(), tk.BooleanVar()
    level_var = tk.IntVar(value=0)

    def pick_dir(var: tk.StringVar, title: str) -> None:
        d = filedialog.askdirectory(title=title)
        if d:
            var.set(d)

    ttk.Label(opt, text="Output folder:").grid(row=0, column=0, sticky="w", **pad)
    ttk.Entry(opt, textvariable=out_var).grid(row=0, column=1, sticky="ew", **pad)
    ttk.Button(opt, text="Browse...", command=lambda: pick_dir(out_var, "Output folder")).grid(row=0, column=2, **pad)
    ttk.Label(opt, text="(empty: next to each video)").grid(row=1, column=1, sticky="w", padx=8)

    fill_widgets: list = []
    ttk.Checkbutton(opt, text="Fill the gaps with ambience from a library", variable=fill_var,
                    command=lambda: refresh_fill()).grid(row=2, column=0, columnspan=3, sticky="w", **pad)
    ttk.Label(opt, text="Library folder:").grid(row=3, column=0, sticky="w", **pad)
    lib_entry = ttk.Entry(opt, textvariable=lib_var)
    lib_entry.grid(row=3, column=1, sticky="ew", **pad)
    lib_btn = ttk.Button(opt, text="Browse...", command=lambda: pick_dir(lib_var, "Ambience library folder"))
    lib_btn.grid(row=3, column=2, **pad)
    ttk.Label(opt, text="Fill level (dB):").grid(row=4, column=0, sticky="w", **pad)
    level_text = ttk.Label(opt, text="0 dB")
    scale = ttk.Scale(opt, from_=-FILL_LEVEL_LIMIT_DB, to=FILL_LEVEL_LIMIT_DB, orient="horizontal",
                      command=lambda v: (level_var.set(round(float(v))), level_text.config(text=f"{level_var.get():+d} dB")))
    scale.set(0)
    scale.grid(row=4, column=1, sticky="ew", **pad)
    level_text.grid(row=4, column=2, **pad)
    fill_widgets = [lib_entry, lib_btn, scale]

    def refresh_fill() -> None:
        for w in fill_widgets:
            w.state(["!disabled"] if fill_var.get() else ["disabled"])
    refresh_fill()

    ttk.Label(opt, text="Keep original:").grid(row=5, column=0, sticky="w", **pad)
    ttk.Entry(opt, textvariable=keep_var).grid(row=5, column=1, columnspan=2, sticky="ew", **pad)
    ttk.Label(opt, text="parts to leave untouched, e.g.  0:00-1:30, 22:10-end").grid(row=6, column=1, sticky="w", padx=8)
    ttk.Checkbutton(opt, text="Also keep the original audio track in the output",
                    variable=orig_var).grid(row=7, column=0, columnspan=3, sticky="w", **pad)

    # --- run
    run = ttk.Frame(root)
    run.pack(fill="x", **pad)
    start_btn = ttk.Button(run, text="Start")
    cancel_btn = ttk.Button(run, text="Cancel", state="disabled")
    open_btn = ttk.Button(run, text="Open output folder", state="disabled")
    start_btn.pack(side="left")
    cancel_btn.pack(side="left", padx=6)
    open_btn.pack(side="left")
    bar = ttk.Progressbar(run, maximum=100)
    bar.pack(side="left", fill="x", expand=True, padx=8)
    status_var = tk.StringVar(value="Add videos, then press Start.")
    ttk.Label(root, textvariable=status_var, wraplength=600).pack(fill="x", padx=8)

    logf = ttk.LabelFrame(root, text="Messages")
    logf.pack(fill="both", expand=True, **pad)
    log = tk.Text(logf, height=7, wrap="word", state="disabled")
    log.tag_configure("error", foreground="#b00020")
    sb = ttk.Scrollbar(logf, command=log.yview)
    log.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    log.pack(fill="both", expand=True)

    def say(text: str, tag: str = "") -> None:
        log.config(state="normal")
        log.insert("end", text + "\n", tag)
        log.see("end")
        log.config(state="disabled")

    def item_for(name: str) -> str | None:
        # the first file of that name that is not finished
        for item, p in rows.items():
            if p.name == name and not tree.set(item, "status").startswith(("done", "failed")):
                return item
        return None

    def set_busy(busy: bool) -> None:
        start_btn.state(["disabled"] if busy else ["!disabled"])
        cancel_btn.state(["!disabled"] if busy else ["disabled"])

    def options() -> Options:
        files = list(rows.values())
        return Options(files=files,
                       out_dir=Path(out_var.get().strip()) if out_var.get().strip() else None,
                       fill=fill_var.get(),
                       library=Path(lib_var.get().strip()) if lib_var.get().strip() else None,
                       fill_level=float(level_var.get()), keep=keep_var.get(), keep_original=orig_var.get())

    def poll() -> None:
        runner: JobRunner | None = state["runner"]
        if runner is None:
            return
        try:
            while True:
                ev = runner.events.get_nowait()
                if ev[0] == "progress":
                    p: Progress = ev[1]
                    item = item_for(p.name)
                    text = p.step if p.percent is None else f"{p.step} {p.percent}%" + (f"  eta {p.eta}" if p.eta else "")
                    if item:
                        set_status(item, "done" if p.step == "done" else text)
                    if p.percent is not None:
                        bar["value"] = p.percent
                    status_var.set(f"File {p.index} of {p.total}: {p.name}  -  {text}")
                elif ev[0] == "line":
                    line = ev[1]
                    state["tail"] = (state["tail"] + [line])[-12:]
                    if line.startswith("failed:") or line.startswith("error:") or "Traceback" in line:
                        state["errors"].append(line)
                        say(line, "error")
                        m = re.match(r"failed: (.*?): ", line)
                        if m:
                            item = item_for(Path(m.group(1)).name)
                            if item:
                                set_status(item, "failed")
                    else:
                        say(line)
                else:
                    finish(ev[1], ev[2])
                    return
        except queue.Empty:
            pass
        root.after(150, poll)

    def finish(code: int, cancelled: bool) -> None:
        state["runner"] = None
        set_busy(False)
        bar["value"] = 0
        for item in rows:
            if tree.set(item, "status") not in ("done", "failed"):
                set_status(item, "cancelled" if cancelled else tree.set(item, "status"))
        folder = state["out_dir"]
        open_btn.state(["!disabled"] if folder and folder.is_dir() else ["disabled"])
        if cancelled:
            status_var.set("Cancelled. A half-finished file is never written.")
        elif code == 0:
            status_var.set("Finished. All files done.")
            if folder:
                open_btn.state(["!disabled"])
        else:
            status_var.set(f"Finished with errors (exit code {code}).")
            detail = "\n".join(state["errors"]) or "\n".join(state["tail"]) or "No message from the command."
            messagebox.showerror("Music remover", f"Something went wrong:\n\n{detail}"[:1500])

    def start() -> None:
        try:
            opts = options()
            cmd = build_command(opts)
        except ValueError as exc:
            messagebox.showwarning("Music remover", str(exc))
            return
        if opts.out_dir is not None:
            state["out_dir"] = opts.out_dir
        else:
            state["out_dir"] = opts.files[0].parent
        state["errors"], state["tail"] = [], []
        for item in rows:
            set_status(item, "waiting")
        say("Started: " + show_command(cmd))
        runner = JobRunner(cmd, PROJECT_ROOT if (PROJECT_ROOT / "models").is_dir() else None)
        state["runner"] = runner
        set_busy(True)
        open_btn.state(["disabled"])
        status_var.set("Starting (loading the model can take a minute)...")
        runner.start()
        root.after(150, poll)

    def cancel() -> None:
        runner = state["runner"]
        if runner is not None and messagebox.askyesno("Music remover", "Stop the running job?"):
            runner.cancel()

    def open_out() -> None:
        folder = state["out_dir"]
        if folder and Path(folder).is_dir():
            _open_folder(Path(folder))

    def on_close() -> None:
        runner = state["runner"]
        if runner is not None:
            if not messagebox.askyesno("Music remover", "A job is running. Stop it and close?"):
                return
            runner.cancel()
        root.destroy()

    start_btn.config(command=start)
    cancel_btn.config(command=cancel)
    open_btn.config(command=open_out)
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
