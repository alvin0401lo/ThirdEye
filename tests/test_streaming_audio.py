import threading
from types import SimpleNamespace

import numpy as np
import pytest

from thirdeye.audio import IncompleteSpeechError, WakeWordListener, _capture_utterance


def test_capture_continues_during_preview_and_only_final_is_returned(monkeypatch, capsys):
    listener = WakeWordListener()
    monkeypatch.setattr(listener, "prepare", lambda: None)
    preview_started = threading.Event()
    release_preview = threading.Event()
    calls = []
    snapshots = np.ones(16_000, dtype=np.int16) * 1000
    complete = np.ones(32_000, dtype=np.int16) * 1000

    def transcribe(samples, hotwords=None):
        calls.append(len(samples))
        if len(calls) == 1:
            preview_started.set()
            assert release_preview.wait(2)
            return "Find my cup"
        return "Find my bottle"

    def capture(*args, on_partial, **kwargs):
        on_partial(snapshots, 16_000)
        assert preview_started.wait(2)
        # Capture remains active even though inference has not completed.
        for _ in range(10):
            on_partial(snapshots, 16_000)
        release_preview.set()
        return complete

    monkeypatch.setattr(listener, "transcribe_samples", transcribe)
    monkeypatch.setattr("thirdeye.audio._capture_utterance", capture)
    assert listener.listen_text() == "Find my bottle"
    assert len(calls) == 2  # One preview, one full final; no queued previews.
    output = capsys.readouterr().out
    assert output.count("[ASR FINAL]") == 1
    assert "[ASR FINAL] Find my bottle" in output
    assert not output.split("[ASR FINAL]", 1)[1].count("[ASR PARTIAL]")


def test_cancel_suppresses_inflight_preview_and_final(monkeypatch, capsys):
    listener = WakeWordListener()
    monkeypatch.setattr(listener, "prepare", lambda: None)
    started = threading.Event()
    release = threading.Event()
    stop = threading.Event()
    calls = []

    def transcribe(samples, hotwords=None):
        calls.append(True)
        started.set()
        assert release.wait(2)
        return "done"

    def capture(*args, on_partial, **kwargs):
        on_partial(np.ones(16_000, dtype=np.int16) * 1000, 16_000)
        assert started.wait(2)
        stop.set()
        release.set()
        return None

    monkeypatch.setattr(listener, "transcribe_samples", transcribe)
    monkeypatch.setattr("thirdeye.audio._capture_utterance", capture)
    assert listener.listen_text(stop_event=stop) is None
    assert len(calls) == 1
    assert "[ASR" not in capsys.readouterr().out


def test_followup_timeout_does_not_transcribe(monkeypatch):
    listener = WakeWordListener()
    monkeypatch.setattr(listener, "prepare", lambda: None)
    monkeypatch.setattr(listener, "transcribe_samples", lambda *args: pytest.fail("Transcribed silence"))
    monkeypatch.setattr("thirdeye.audio._capture_utterance", lambda *args, **kwargs: None)
    assert listener.listen_text(wait_timeout=0.1) is None


def test_pcm_input_is_normalized_for_whisper():
    calls = []
    listener = WakeWordListener()
    listener._model = SimpleNamespace(transcribe=lambda samples, **kwargs: (calls.append(samples) or iter([SimpleNamespace(text="hello")]), None))
    assert listener.transcribe_samples(np.array([-32768, 0, 16384], dtype=np.int16)) == "hello"
    np.testing.assert_array_equal(calls[0], np.array([-1, 0, 0.5], dtype=np.float32))


def test_capture_emits_snapshots_and_keeps_preroll(monkeypatch):
    chunks = [np.zeros((800, 1), dtype=np.int16) for _ in range(12)]
    chunks += [np.full((800, 1), 1000, dtype=np.int16) for _ in range(24)]
    chunks += [np.zeros((800, 1), dtype=np.int16) for _ in range(9)]
    snapshots = []
    class Stream:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, count): return chunks.pop(0), False
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", SimpleNamespace(
        query_devices=lambda *args: {"default_samplerate": 16000}, InputStream=lambda **kwargs: Stream(),
    ))
    result = _capture_utterance(None, on_partial=lambda samples, rate: snapshots.append(samples), require_complete=True)
    assert result is not None
    assert snapshots
    assert np.all(snapshots[0][:8800] == 0)
    assert np.any(snapshots[0][8800:] != 0)


@pytest.mark.parametrize("overflow", [True, False])
def test_incomplete_recording_is_never_finalized(monkeypatch, overflow):
    now = [0.0]
    class Stream:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, count):
            now[0] += 0.05
            return np.full((800, 1), 1000, dtype=np.int16), overflow
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", SimpleNamespace(
        query_devices=lambda *args: {"default_samplerate": 16000}, InputStream=lambda **kwargs: Stream(),
    ))
    monkeypatch.setattr("thirdeye.audio.time.monotonic", lambda: now[0])
    with pytest.raises(IncompleteSpeechError):
        _capture_utterance(None, duration=0.3, require_complete=True)


def test_wake_matching_ignores_provisional_text(monkeypatch, capsys):
    listener = WakeWordListener()
    stop = threading.Event()
    def listen(*args, **kwargs):
        print("[ASR PARTIAL] hi third eye")
        stop.set()
        return "hello there"
    monkeypatch.setattr(listener, "listen_text", listen)
    assert listener.wait_for("hi third eye", stop) is None
