import sys
from pathlib import Path

import pytest

from musicremover import gui
from musicremover.gui import (JobRunner, Options, build_args, expand_inputs, parse_progress,
                              show_command, split_keep, split_lines)


def test_minimal_args_are_just_the_files():
    assert build_args(Options(files=[Path("a.mkv"), Path("b.mkv")])) == ["a.mkv", "b.mkv"]


def test_no_files_is_an_error():
    with pytest.raises(ValueError, match="at least one"):
        build_args(Options())


def test_paths_with_spaces_stay_single_arguments_and_are_quoted_for_display():
    p = Path("D:/My Anime/Episode 01.mkv")
    out = Path("D:/Out Folder")
    args = build_args(Options(files=[p], out_dir=out))
    assert args == [str(p), "-o", str(out)]
    shown = show_command(["python", *args])
    assert f'"{p}"' in shown and f'"{out}"' in shown


def test_split_keep_variants():
    assert split_keep("0:00-1:30, 22:10-end") == ["0:00-1:30", "22:10-end"]
    assert split_keep(" 0:00-1:30 ;22:10-end\n5:00-6:00") == ["0:00-1:30", "22:10-end", "5:00-6:00"]
    assert split_keep("") == [] and split_keep(" , ") == []


def test_split_keep_rejects_bad_range_naming_it():
    with pytest.raises(ValueError, match="1:30-0:10"):
        split_keep("0:00-1:00, 1:30-0:10")
    with pytest.raises(ValueError, match="keep range"):
        split_keep("banana")


def test_keep_becomes_repeated_option():
    args = build_args(Options(files=[Path("a.mkv")], keep="0:00-1:30, 22:10-end", keep_original=True))
    assert args == ["a.mkv", "--keep", "0:00-1:30", "--keep", "22:10-end", "--keep-original"]


def test_fill_needs_library_and_level_only_when_nonzero():
    with pytest.raises(ValueError, match="library"):
        build_args(Options(files=[Path("a.mkv")], fill=True))
    lib = Path("D:/Sound Library")
    base = ["a.mkv", "--fill", "library", "--library", str(lib)]
    assert build_args(Options(files=[Path("a.mkv")], fill=True, library=lib)) == base
    assert build_args(Options(files=[Path("a.mkv")], fill=True, library=lib, fill_level=-6)) == base + ["--fill-level", "-6"]
    assert build_args(Options(files=[Path("a.mkv")], fill=True, library=lib, fill_level=99))[-1] == "20"


def test_fill_off_ignores_library_and_level():
    args = build_args(Options(files=[Path("a.mkv")], fill=False, library=Path("L"), fill_level=5))
    assert args == ["a.mkv"]


def test_device_passed_through():
    assert build_args(Options(files=[Path("a.mkv")], device="cpu"))[-2:] == ["--device", "cpu"]


def test_parse_progress_with_percent_and_eta():
    p = parse_progress("[1/3] My Show - 01.mkv  separating 12%  eta 5:20")
    assert (p.index, p.total, p.name, p.step, p.percent, p.eta) == (1, 3, "My Show - 01.mkv", "separating", 12, "5:20")


def test_parse_progress_multiword_step_and_plain_steps():
    p = parse_progress("[2/3] a.mkv  rescuing voices 100%  eta 0:00")
    assert (p.step, p.percent) == ("rescuing voices", 100)
    p = parse_progress("[2/3] a.mkv  extracting")
    assert (p.step, p.percent, p.eta) == ("extracting", None, None)
    assert parse_progress("[2/3] a.mkv  done").step == "done"


def test_parse_progress_ignores_other_lines():
    assert parse_progress("D:\\out\\a.nomusic.mkv  (5:01)") is None
    assert parse_progress("failed: a.mkv: boom") is None
    assert parse_progress("  ambience laid under 0:30 in 2 places") is None


def test_split_lines_handles_cr_updates_and_partial_tail():
    lines, tail = split_lines(b"[1/1] a  separating 1%\r[1/1] a  separating 2%\r[1/1] a  sep")
    assert lines == ["[1/1] a  separating 1%", "[1/1] a  separating 2%"] and tail == b"[1/1] a  sep"
    assert split_lines(b"x\r\ny\n")[0] == ["x", "y"]


def test_expand_inputs_takes_videos_from_folders(tmp_path):
    for n in ("b.mkv", "a.mp4", "c.txt", "a.nomusic.mkv"):
        (tmp_path / n).write_bytes(b"")
    f = tmp_path.parent / (tmp_path.name + "-x.mkv")
    f.write_bytes(b"")
    assert expand_inputs([tmp_path, f]) == [tmp_path / "a.mp4", tmp_path / "b.mkv", f]


def test_build_command_uses_module():
    cmd = gui.build_command(Options(files=[Path("a.mkv")]))
    assert cmd[1:3] == ["-m", "musicremover"] and cmd[-1] == "a.mkv"


FAKE = r"""
import sys
sys.stderr.write("\r[1/1] clip.mkv  extracting")
sys.stderr.flush()
for i in (10, 50, 100):
    sys.stderr.write("\r[1/1] clip.mkv  separating %d%%  eta 0:0%d" % (i, i // 50))
    sys.stderr.flush()
sys.stderr.write("\r[1/1] clip.mkv  done\n")
print("out.mkv  (0:01)")
sys.exit(1 if "--fail" in sys.argv else 0)
"""


def _drain(runner):
    runner.wait(20)
    events = []
    while not runner.events.empty():
        events.append(runner.events.get())
    return events


def test_job_runner_reports_progress_lines_and_exit(tmp_path):
    script = tmp_path / "fake.py"
    script.write_text(FAKE)
    r = JobRunner([sys.executable, str(script)])
    r.start()
    ev = _drain(r)
    prog = [e[1] for e in ev if e[0] == "progress"]
    assert prog[0].step == "extracting" and prog[-1].step == "done"
    assert any(p.percent == 50 and p.eta == "0:01" for p in prog)
    assert ("line", "out.mkv  (0:01)") in ev
    assert ev[-1] == ("done", 0, False)


def test_job_runner_exit_code_and_missing_command(tmp_path):
    script = tmp_path / "fake.py"
    script.write_text(FAKE)
    r = JobRunner([sys.executable, str(script), "--fail"])
    r.start()
    assert _drain(r)[-1][:2] == ("done", 1)
    r = JobRunner([str(tmp_path / "nonexistent.exe")])
    r.start()
    ev = _drain(r)
    assert ev[0][0] == "line" and ev[-1][0] == "done" and ev[-1][1] != 0


def test_job_runner_cancel_stops_the_process(tmp_path):
    script = tmp_path / "slow.py"
    script.write_text("import time\nprint('start', flush=True)\ntime.sleep(60)\n")
    r = JobRunner([sys.executable, str(script)])
    r.start()
    assert r.events.get(timeout=20)[0] == "line"
    assert r.running()
    r.cancel()
    ev = _drain(r)
    assert not r.running()
    assert ev[-1][0] == "done" and ev[-1][2] is True
