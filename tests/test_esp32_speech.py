import threading
import time
import wave
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from thirdeye.app import ThirdEyeApp
from thirdeye.audio import IncompleteSpeechError, SpeechHighPass
from thirdeye.camera import CameraError


class FakeMicrophone:
    def __init__(self, chunks, stop_event=None):
        self.chunks = iter(chunks)
        self.stop_event = stop_event

    def read_audio(self, timeout=1.0):
        try:
            return next(self.chunks)
        except StopIteration:
            if self.stop_event is not None:
                self.stop_event.set()
            raise CameraError("No audio")


def _pcm(value):
    # Voice-like AC, not a constant voltage that the high-pass should reject.
    return (value * np.sin(2 * np.pi * 400 * np.arange(320) / 16000)).astype(np.int16).tobytes()


def test_esp32_speech_ends_after_silence_and_keeps_preroll():
    app = object.__new__(ThirdEyeApp)
    app._esp32_voice_thresholds = (500.0, 80.0)
    app._network_camera = FakeMicrophone([_pcm(0)] * 5 + [_pcm(2000)] * 12 + [_pcm(0)] * 26)

    path = app._record_esp32_audio(4.0)
    assert path is not None
    try:
        with wave.open(str(path), "rb") as audio:
            assert audio.getframerate() == 16_000
            assert 0.55 < audio.getnframes() / 16_000 < 1.0
    finally:
        path.unlink(missing_ok=True)


def test_high_pass_removes_dc_and_preserves_packet_continuity():
    signal = (2000 + 300 * np.sin(2 * np.pi * 400 * np.arange(16000) / 16000)).astype(np.int16)
    whole = SpeechHighPass().process(signal)
    streaming = SpeechHighPass()
    packets = np.concatenate([streaming.process(chunk) for chunk in np.array_split(signal, 50)])
    assert np.array_equal(whole, packets)
    assert abs(float(whole[320:].mean())) < 2
    assert 180 < float(np.sqrt(np.mean(whole[320:].astype(float) ** 2))) < 220
    assert not SpeechHighPass().process(np.full(320, 2000, dtype=np.int16)).any()


def test_esp32_does_not_wake_on_dc_drift_alone():
    stop = threading.Event()
    app = object.__new__(ThirdEyeApp)
    app._esp32_voice_thresholds = (500.0, 80.0)
    app._network_camera = FakeMicrophone([np.full(320, 2000, dtype=np.int16).tobytes()] * 50, stop)
    assert app._record_esp32_audio(4.0, stop) is None


def test_esp32_speech_ignores_short_noise_and_can_stop():
    stop_event = threading.Event()
    app = object.__new__(ThirdEyeApp)
    app._esp32_voice_thresholds = (500.0, 80.0)
    app._network_camera = FakeMicrophone(
        [_pcm(2000)] * 2 + [_pcm(0)] * 23,
        stop_event,
    )

    assert app._record_esp32_audio(2.0, stop_event) is None


@pytest.mark.parametrize("minimum,voice_packets,accepted", [(0.20, 7, False), (0.12, 7, True), (0.12, 2, False)])
def test_short_completion_threshold_preserves_noise_rejection(minimum, voice_packets, accepted):
    stop = threading.Event()
    app = object.__new__(ThirdEyeApp)
    app._esp32_voice_thresholds = (500.0, 80.0)
    app._network_camera = FakeMicrophone([_pcm(2000)] * voice_packets + [_pcm(0)] * 26, stop)
    samples = app._record_esp32_audio(2.0, stop, min_voiced_seconds=minimum, return_samples=True)
    assert (samples is not None) == accepted


def test_audio_stall_during_speech_is_an_incomplete_command(monkeypatch):
    clock = [0.0]
    class StalledMicrophone:
        def read_audio(self, timeout=1.0):
            if clock[0] == 0:
                clock[0] = 0.1
                return _pcm(2000)
            clock[0] += 2
            raise CameraError("No audio")
    monkeypatch.setattr("thirdeye.app.time.monotonic", lambda: clock[0])
    app = object.__new__(ThirdEyeApp)
    app._esp32_voice_thresholds = (500.0, 80.0)
    app._network_camera = StalledMicrophone()
    with pytest.raises(IncompleteSpeechError, match="stopped during speech"):
        app._record_esp32_audio(4.0, require_complete=True)


@pytest.mark.parametrize("phrase,options", [("done", {"min_voiced_seconds": 0.12}), ("hi third eye", {})])
def test_only_done_uses_short_completion_threshold(tmp_path, phrase, options):
    path = tmp_path / "word.wav"
    path.write_bytes(b"speech")
    app = object.__new__(ThirdEyeApp)
    def record(seconds, stop_event, **kwargs):
        assert kwargs == options
        return path
    app._record_esp32_audio = record
    listener = SimpleNamespace(matches_clip=lambda *args, **kwargs: True)
    assert app._wait_for_esp32_phrase(listener, phrase, 2.0)


def test_esp32_short_noise_does_not_abort_streaming_asr():
    app = object.__new__(ThirdEyeApp)
    app._esp32_voice_thresholds = (500.0, 80.0)
    app._network_camera = FakeMicrophone(
        [_pcm(0)] * 5 + [_pcm(2000)] * 2 + [_pcm(0)] * 25
        + [_pcm(2000)] * 12 + [_pcm(0)] * 26
    )
    streamed = []

    samples = app._record_esp32_audio(
        4.0, on_chunk=lambda chunk, rate: streamed.append((chunk.copy(), rate)),
        require_complete=True, return_samples=True,
    )

    assert samples.size > 0
    assert streamed
    assert all(rate == 16_000 for _, rate in streamed)
    assert sum(len(chunk) for chunk, _ in streamed) == samples.size


def test_esp32_speech_streams_pcm_before_utterance_finishes():
    app = object.__new__(ThirdEyeApp)
    app._esp32_voice_thresholds = (500.0, 80.0)
    app._network_camera = FakeMicrophone([_pcm(0)] * 5 + [_pcm(2000)] * 12 + [_pcm(0)] * 26)
    seen = []
    samples = app._record_esp32_audio(
        4.0, on_chunk=lambda chunk, rate: seen.append((len(chunk), rate)),
        require_complete=True, return_samples=True,
    )
    assert samples.size > 0
    assert len(seen) > 1
    assert sum(length for length, _ in seen) == samples.size
    assert all(rate == 16_000 for _, rate in seen)


def test_esp32_realtime_discards_unfinished_speech():
    app = object.__new__(ThirdEyeApp)
    app._esp32_voice_thresholds = (500.0, 80.0)
    app._network_camera = FakeMicrophone([_pcm(2000)] * 120)
    with pytest.raises(IncompleteSpeechError, match="time limit"):
        app._record_esp32_audio(
            1.0, on_chunk=lambda chunk, rate: None,
            require_complete=True, return_samples=True,
        )


def test_esp32_realtime_discards_audio_gap_during_speech():
    class GapMicrophone:
        def __init__(self):
            self.reads = 0
            self.faults = 0

        @property
        def audio_integrity(self):
            return "sequenced", self.faults, self.faults, 0

        def read_audio(self, timeout=1.0):
            self.reads += 1
            if self.reads == 6:
                self.faults += 1
            return _pcm(2000)

    app = object.__new__(ThirdEyeApp)
    app._esp32_voice_thresholds = (500.0, 80.0)
    app._network_camera = GapMicrophone()
    with pytest.raises(IncompleteSpeechError, match="audio gap"):
        app._record_esp32_audio(
            4.0, on_chunk=lambda chunk, rate: None,
            require_complete=True, return_samples=True,
        )


def test_esp32_realtime_requires_sequenced_firmware():
    class LegacyMicrophone:
        audio_integrity = ("legacy", 0, 0, 0)

        def read_audio(self, timeout=1.0):
            return _pcm(2000)

    app = object.__new__(ThirdEyeApp)
    app._esp32_voice_thresholds = (500.0, 80.0)
    app._network_camera = LegacyMicrophone()
    with pytest.raises(CameraError, match="sequence numbers"):
        app._record_esp32_audio(4.0, require_complete=True, return_samples=True)


def test_esp32_wake_retries_after_audio_gap(tmp_path):
    path = tmp_path / "wake.wav"
    path.write_bytes(b"speech")
    calls = []
    app = object.__new__(ThirdEyeApp)

    def record(seconds, stop_event):
        calls.append(seconds)
        if len(calls) == 1:
            raise IncompleteSpeechError("audio gap")
        return path

    app._record_esp32_audio = record
    listener = SimpleNamespace(matches_clip=lambda *args, **kwargs: True)
    assert app._wait_for_esp32_phrase(listener, "hi third eye", 4.0)
    assert calls == [4.0, 4.0]


def test_esp32_microphone_calibrates_noise_before_detection():
    app = object.__new__(ThirdEyeApp)
    app._esp32_voice_thresholds = None
    app._network_camera = FakeMicrophone([_pcm(200)] * 40)

    assert app._calibrate_esp32_microphone()
    assert app._esp32_voice_thresholds[0] == 500.0
    assert 350 < app._esp32_voice_thresholds[1] < 450


def test_esp32_microphone_rejects_speech_during_calibration():
    app = object.__new__(ThirdEyeApp)
    app._esp32_voice_thresholds = None
    app._network_camera = FakeMicrophone([_pcm(2000)] * 40)

    with pytest.raises(CameraError, match="calibration too loud"):
        app._calibrate_esp32_microphone()


def test_esp32_microphone_test_writes_raw_audio(capsys):
    class StreamingMicrophone:
        def clear_audio_buffer(self):
            pass

        def read_audio(self, timeout=0.25):
            time.sleep(0.01)
            return _pcm(1000)

    app = object.__new__(ThirdEyeApp)
    app.settings = SimpleNamespace(camera_source="esp32")
    app._network_camera = StreamingMicrophone()
    app._open_camera = lambda: nullcontext()

    app.test_esp32_microphone(seconds=0.03)
    output = capsys.readouterr().out
    path = Path(output.split("Raw microphone recording: ")[1].strip())
    try:
        with wave.open(str(path), "rb") as audio:
            assert audio.getnframes() > 0
        assert "clipped=0.0%" in output
    finally:
        path.unlink(missing_ok=True)


def test_esp32_microphone_test_reports_connected_without_packets():
    class SilentMicrophone:
        audio_connected = True

        def clear_audio_buffer(self):
            pass

        def read_audio(self, timeout=15.0):
            raise CameraError("No audio")

    app = object.__new__(ThirdEyeApp)
    app.settings = SimpleNamespace(camera_source="esp32")
    app._network_camera = SilentMicrophone()
    app._open_camera = lambda: nullcontext()

    with pytest.raises(CameraError, match="connected, but no audio packets arrived"):
        app.test_esp32_microphone(seconds=0.03)
