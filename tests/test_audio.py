import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from thirdeye.audio import (
    WakeWordListener,
    _capture_utterance,
    _phrase_match,
    _prepare_speech_samples,
    _resample_audio,
)


def test_wake_word_listener_uses_local_whisper(monkeypatch) -> None:
    class FakeModel:
        def transcribe(self, path, **kwargs):
            return iter([SimpleNamespace(text=" Hi, Third Eye.")]), None

    listener = WakeWordListener(minimum_confidence=0.70)
    listener._model = FakeModel()
    monkeypatch.setattr(
        listener,
        "listen_text",
        lambda *args, **kwargs: "Hi, Third Eye.",
    )

    assert listener.wait() == "hi third eye"


def test_wake_transcribes_once_without_provisional_model_work(monkeypatch) -> None:
    listener = WakeWordListener()
    samples = np.full(16_000, 1000, dtype=np.int16)
    calls = []

    monkeypatch.setattr(listener, "prepare", lambda: None)

    def capture(*args, **kwargs):
        assert "on_partial" not in kwargs
        return samples

    def transcribe(audio, hotwords):
        calls.append((audio, hotwords))
        return "Hi Third Eye"

    monkeypatch.setattr("thirdeye.audio._capture_utterance", capture)
    monkeypatch.setattr(listener, "transcribe_samples", transcribe)

    assert listener.wait() == "hi third eye"
    assert len(calls) == 1
    assert calls[0][1] == "hi third eye"


def test_command_transcription_reuses_loaded_wake_model() -> None:
    calls = []

    class FakeModel:
        def transcribe(self, path, **kwargs):
            calls.append((path, kwargs))
            return iter([SimpleNamespace(text=" Find my bottle.")]), None

    listener = WakeWordListener(model_name="small.en")
    listener._model = FakeModel()

    assert listener.transcribe_clip(Path("command.wav")) == "Find my bottle."
    assert calls[0][1]["beam_size"] == 1
    assert calls[0][1]["hotwords"] is None


def test_done_requires_a_complete_word() -> None:
    assert _phrase_match("done", "done", 1.0)[0]
    assert _phrase_match("I am done", "done", 1.0)[0]
    assert not _phrase_match("don't", "done", 1.0)[0]
    assert not _phrase_match("undone", "done", 1.0)[0]


def test_wake_phrase_accepts_a_high_confidence_adjacent_partial() -> None:
    assert _phrase_match("hi third", "hi third eye", 0.70, allow_wake_partial=True)[0]
    assert _phrase_match("third eye", "hi third eye", 0.70, allow_wake_partial=True)[0]
    assert not _phrase_match("hi there", "hi third eye", 0.70, allow_wake_partial=True)[0]


def test_phrase_confidence_override_does_not_change_wake_threshold(monkeypatch) -> None:
    class FakeModel:
        def transcribe(self, path, **kwargs):
            return iter([SimpleNamespace(text="done")]), None

    listener = WakeWordListener(minimum_confidence=0.70)
    listener._model = FakeModel()
    monkeypatch.setattr(
        listener,
        "listen_text",
        lambda *args, **kwargs: "done",
    )

    assert listener.wait_for("done", minimum_confidence=1.0) == "done"
    assert listener.minimum_confidence == 0.70


def test_prepare_speech_trims_silence_and_amplifies_quiet_voice() -> None:
    sample_rate = 16_000
    silence = np.zeros(sample_rate, dtype=np.int16)
    speech = np.full(sample_rate // 2, 1_000, dtype=np.int16)

    prepared = _prepare_speech_samples(np.concatenate((silence, speech, silence)), sample_rate)

    assert len(prepared) < len(silence) * 2
    assert prepared.max() > speech.max()


def test_resample_audio_matches_output_rate() -> None:
    samples = np.arange(22_050, dtype=np.int16)

    resampled = _resample_audio(samples, 22_050, 48_000)

    assert len(resampled) == 48_000


@pytest.mark.parametrize(("end_silence_seconds", "silent_chunks"), [(0.45, 9), (0.25, 5)])
def test_capture_stops_after_speech_and_trailing_silence(monkeypatch, end_silence_seconds, silent_chunks) -> None:
    sample_rate = 16_000
    chunk_size = 800
    chunks = [np.zeros((chunk_size, 1), dtype=np.int16) for _ in range(2)]
    chunks.extend(np.full((chunk_size, 1), 2_000, dtype=np.int16) for _ in range(5))
    chunks.extend(np.zeros((chunk_size, 1), dtype=np.int16) for _ in range(silent_chunks))

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _frames):
            return chunks.pop(0), False

    fake_sounddevice = SimpleNamespace(
        query_devices=lambda *_args: {"default_samplerate": sample_rate},
        InputStream=lambda **_kwargs: FakeStream(),
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    captured = _capture_utterance(
        None, sample_rate=sample_rate, duration=4.0, end_silence_seconds=end_silence_seconds,
    )

    assert captured is not None
    assert len(captured) < sample_rate * 2


def test_capture_followup_timeout_returns_without_speech(monkeypatch) -> None:
    sample_rate = 16_000
    now = [0.0]

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, frames):
            now[0] += 0.05
            return np.zeros((frames, 1), dtype=np.int16), False

    fake_sounddevice = SimpleNamespace(
        query_devices=lambda *_args: {"default_samplerate": sample_rate},
        InputStream=lambda **_kwargs: FakeStream(),
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)
    monkeypatch.setattr("thirdeye.audio.time.monotonic", lambda: now[0])

    assert _capture_utterance(None, wait_timeout=0.15) is None
