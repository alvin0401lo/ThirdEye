import sys
import shutil
from pathlib import Path
from types import SimpleNamespace

from thirdeye.cue_audio import CUE_FILES, CuePlayer


def test_object_not_found_has_a_dedicated_cue_file():
    assert CUE_FILES["object not found"] == "object_not_found.wav"


def _test_audio_dir(name: str) -> Path:
    path = Path(__file__).parent / "_tmp_cue_audio" / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def test_cue_player_uses_wav_when_file_exists(monkeypatch):
    calls = []
    audio_dir = _test_audio_dir("wav")
    (audio_dir / "left.wav").write_bytes(b"RIFF")
    monkeypatch.setitem(
        sys.modules,
        "winsound",
        SimpleNamespace(
            SND_FILENAME=1,
            PlaySound=lambda path, flags: calls.append(("wav", path, flags)),
            Beep=lambda frequency, duration: calls.append(("beep", frequency, duration)),
        ),
    )
    player = CuePlayer(audio_dir)
    try:
        player.play("Left")
        player.wait()
    finally:
        player.close()
        shutil.rmtree(audio_dir.parent, ignore_errors=True)

    assert calls == [("wav", str(audio_dir / "left.wav"), 1)]


def test_cue_player_keeps_only_latest_waiting_cue(monkeypatch):
    calls = []
    release = []
    audio_dir = _test_audio_dir("latest")

    def fake_beep(frequency, duration):
        calls.append((frequency, duration))
        if not release:
            while not release:
                pass

    monkeypatch.setitem(
        sys.modules,
        "winsound",
        SimpleNamespace(SND_FILENAME=1, PlaySound=lambda *_args: None, Beep=fake_beep),
    )
    player = CuePlayer(audio_dir)
    try:
        player.play("Left")
        player.play("Right")
        player.play("Up")
        release.append(True)
        player.wait()
    finally:
        player.close()
        shutil.rmtree(audio_dir.parent, ignore_errors=True)

    assert calls[-1] == (1040, 90)
    assert len(calls) <= 2
