import threading
from contextlib import nullcontext
import time
from types import SimpleNamespace

import cv2
import numpy as np

from thirdeye.app import ThirdEyeApp
from thirdeye.audio import IncompleteSpeechError
from thirdeye.models import Command, FramePacket, Intent
from thirdeye.repo_object_finding import RepoGuidance


def test_esp32_realtime_keeps_original_command_and_followup_flow(monkeypatch):
    spoken = []
    questions = []
    timeouts = []
    clears = []
    wake_calls = []

    class FakeWake:
        def __init__(self, *args):
            pass

        def prepare(self):
            pass

    class FakeRealtime:
        def __init__(self, key, device=None, capture_utterance=None):
            assert key == "test-key"
            assert capture_utterance is not None
            self.capture_utterance = capture_utterance
            self.transcripts = iter(["What is in front of me?", "done"])

        def prepare(self):
            pass

        def close(self):
            pass

        def listen_text(self, *, wait_timeout=None):
            timeouts.append(wait_timeout)
            streamed = []
            result = self.capture_utterance(
                None, duration=12.0, wait_timeout=wait_timeout,
                stop_event=threading.Event(), require_complete=True,
                on_chunk=lambda samples, rate: streamed.append((len(samples), rate)),
            )
            assert result is not None
            assert streamed == [(320, 16_000)]
            return next(self.transcripts)

    app = object.__new__(ThirdEyeApp)
    app.settings = SimpleNamespace(
        camera_source="esp32", audio_source="esp32", transcribe_model="gpt-live-transcribe",
        openai_api_key="test-key", microphone_device=None, wake_word="hi third eye",
        wake_language="en-US", wake_confidence=0.7, wake_model="small.en",
    )
    app.ai = SimpleNamespace(
        online=True, understand_command=lambda text: Command(Intent.VISUAL_QUESTION),
    )
    app._network_camera = SimpleNamespace(clear_audio_buffer=lambda: clears.append(True))
    app._preload_realtime_navigation = lambda: None
    app._speak_to_user = spoken.append
    app.answer_visual_question = questions.append

    def record(seconds, stop_event, wait_timeout, *, on_chunk, require_complete, return_samples):
        assert seconds == 12.0 and require_complete and return_samples
        on_chunk(np.full(320, 1000, dtype=np.int16), 16_000)
        return np.full(320, 1000, dtype=np.int16)

    def wake(listener, phrase, seconds, **kwargs):
        wake_calls.append(phrase)
        return len(wake_calls) == 1

    app._record_esp32_audio = record
    app._wait_for_esp32_phrase = wake
    monkeypatch.setattr("thirdeye.app.WakeWordListener", FakeWake)
    monkeypatch.setattr("thirdeye.app.RealtimeTranscriber", FakeRealtime)

    app.run_voice_control()

    assert questions == ["What is in front of me?"]
    assert timeouts == [None, 30.0]
    assert spoken == ["Hi.", "Standing by"]
    assert wake_calls == ["hi third eye", "hi third eye"]
    assert len(clears) >= 2


def test_esp32_audio_gap_retries_command_without_wake_or_spoken_error(monkeypatch, capsys):
    spoken = []
    wake_calls = []
    wait_timeouts = []

    class FakeWake:
        def __init__(self, *args):
            pass

        def prepare(self):
            pass

    class FakeRealtime:
        def __init__(self, *args, **kwargs):
            self.attempts = 0

        def prepare(self):
            pass

        def listen_text(self, *, wait_timeout=None):
            wait_timeouts.append(wait_timeout)
            self.attempts += 1
            if self.attempts == 1:
                raise IncompleteSpeechError("ESP32 microphone audio gap; incomplete command discarded")
            return "done"

    app = object.__new__(ThirdEyeApp)
    app.settings = SimpleNamespace(
        camera_source="esp32", audio_source="esp32", transcribe_model="gpt-live-transcribe",
        openai_api_key="test-key", microphone_device=None, wake_word="hi third eye",
        wake_language="en-US", wake_confidence=0.7, wake_model="small.en",
    )
    app.ai = SimpleNamespace(online=True)
    app._network_camera = SimpleNamespace(clear_audio_buffer=lambda: None)
    app._preload_realtime_navigation = lambda: None
    app._speak_to_user = spoken.append

    def wake(*args, **kwargs):
        wake_calls.append(True)
        return len(wake_calls) == 1

    app._wait_for_esp32_phrase = wake
    monkeypatch.setattr("thirdeye.app.WakeWordListener", FakeWake)
    monkeypatch.setattr("thirdeye.app.RealtimeTranscriber", FakeRealtime)

    app.run_voice_control()

    assert spoken == ["Hi.", "Standing by"]
    assert wait_timeouts == [None, 30.0]
    assert len(wake_calls) == 2
    assert "Audio interrupted; listening again." in capsys.readouterr().out


def test_esp32_voice_find_done_returns_to_wake_mode(tmp_path, monkeypatch):
    spoken = []
    cues = []
    processed = threading.Event()
    release_analysis = threading.Event()
    guidance_shown = threading.Event()
    displayed_during_analysis = []
    audio_path = tmp_path / "command.wav"
    audio_path.write_bytes(b"speech")

    class FakeCameraWorker:
        def __init__(self):
            self.frame_id = 0

        def read(self, *args, **kwargs):
            time.sleep(0.001)
            self.frame_id += 1
            return FramePacket(self.frame_id, time.monotonic(), np.zeros((8, 8, 3), dtype=np.uint8), "esp32")

    class FakeFinder:
        def __init__(self, detector, hand_tracker):
            pass

        def set_target(self, target):
            assert target == "bottle"

        def process(self, frame):
            processed.set()
            assert release_analysis.wait(timeout=2)
            return SimpleNamespace(guidance=RepoGuidance.LEFT)

    app = object.__new__(ThirdEyeApp)
    app.settings = SimpleNamespace(
        camera_source="esp32", wake_word="hi third eye", wake_language="en-US",
        wake_confidence=0.7, microphone_device=None, wake_model="small.en",
        camera_timeout=0.1, guidance_repeat_s=1.5, transcribe_model="local",
    )
    app.ai = SimpleNamespace(
        online=True,
        transcribe=lambda path: (_ for _ in ()).throw(AssertionError("Cloud transcription used")),
        understand_command=lambda text: Command(Intent.FIND_OBJECT, "bottle"),
    )
    app._network_camera = SimpleNamespace(
        clear_audio_buffer=lambda: None,
        video_session=lambda profile="live": nullcontext(),
    )
    app._device_speaker = SimpleNamespace(say=spoken.append)
    def play_cue(cue):
        cues.append(cue)
        guidance_shown.set()
    app.cue_player = SimpleNamespace(play=play_cue)
    app._preload_realtime_navigation = lambda: None
    app._load_detector = lambda: object()
    app._load_hand_tracker = lambda: object()
    app._start_camera_worker = lambda: FakeCameraWorker()
    app._record_esp32_audio = lambda seconds: audio_path
    wake_calls = []

    def fake_wait(listener, phrase, seconds, **_kwargs):
        wake_calls.append(phrase)
        return len(wake_calls) == 1

    def fake_done(stop_event, completed_event):
        assert guidance_shown.wait(timeout=2)
        completed_event.set()

    app._wait_for_esp32_phrase = fake_wait
    app._listen_for_esp32_done = fake_done
    monkeypatch.setattr("thirdeye.app.WakeWordListener.transcribe_clip", lambda self, path: "Find my bottle")
    monkeypatch.setattr("thirdeye.app.WakeWordListener.prepare", lambda self: None)
    monkeypatch.setattr("thirdeye.app.RepoObjectFinder", FakeFinder)
    monkeypatch.setattr("thirdeye.repo_display.annotate_repo_finding", lambda image, result, target: image)
    def display(*args):
        if processed.is_set() and not release_analysis.is_set():
            displayed_during_analysis.append(True)
            if len(displayed_during_analysis) == 3:
                release_analysis.set()
    monkeypatch.setattr(cv2, "imshow", display)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: -1)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    app.run_voice_control()

    assert spoken == [
        "Hi.",
        "Finding bottle. Say done to finish.",
        "Done. Object finding complete.",
    ]
    assert "Left" in cues
    assert len(displayed_during_analysis) == 3
    assert wake_calls == ["hi third eye", "hi third eye"]
