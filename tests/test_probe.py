import dataclasses

import pytest

from musicremover.probe import ProbeError, choose_track, probe


def test_probe_fields(media):
    info = probe(media)
    assert info.has_video
    assert info.subtitle_count == 1
    assert info.duration == pytest.approx(3.0, abs=0.2)
    assert len(info.audio) == 2
    a, b = info.audio
    assert (a.position, b.position) == (0, 1)
    assert a.index == 1 and b.index == 2
    assert (a.channels, a.sample_rate, a.language) == (2, 48000, "jpn")
    assert (b.channels, b.sample_rate, b.language) == (6, 44100, "eng")
    assert a.default and not b.default


def test_frozen(media):
    info = probe(media)
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.duration = 1.0  # type: ignore[misc]


def test_choose_by_language(media):
    info = probe(media)
    assert choose_track(info, "eng").position == 1
    assert choose_track(info, "jpn").position == 0


def test_choose_default(media):
    assert choose_track(probe(media)).position == 0


def test_choose_missing_language(media):
    with pytest.raises(ProbeError, match="eng.*jpn"):
        choose_track(probe(media), "fra")


def test_missing_file(tmp_path):
    with pytest.raises(ProbeError, match="not found"):
        probe(tmp_path / "nope.mkv")


def test_no_audio(video_only):
    with pytest.raises(ProbeError, match="audio"):
        probe(video_only)


def test_not_media(tmp_path):
    f = tmp_path / "x.mkv"
    f.write_text("not media")
    with pytest.raises(ProbeError):
        probe(f)
