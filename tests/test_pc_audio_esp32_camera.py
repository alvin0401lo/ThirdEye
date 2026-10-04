from contextlib import nullcontext
from types import SimpleNamespace
import threading

import pytest

from thirdeye.app import ThirdEyeApp, _listen_for_done
from thirdeye.models import Command, Intent


def test_done_listener_prepares_early_but_waits_for_prompt(monkeypatch):
    ready = threading.Event()
    start = threading.Event()
    stop = threading.Event()
    completed = threading.Event()
    calls = []

    class FakeListener:
        def __init__(self, *args):
            pass

        def prepare(self):
            ready.set()

        def wait_for(self, *args, **kwargs):
            calls.append(kwargs)
            return "done"

    monkeypatch.setattr("thirdeye.app.WakeWordListener", FakeListener)
    worker = threading.Thread(target=_listen_for_done, args=("en-US", None, "small.en", stop, completed, start))
    worker.start()
    assert ready.wait(1)
    assert calls == []
    start.set()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert completed.is_set()
    assert calls[0]["window_seconds"] == 1.5


def test_pc_audio_does_not_route_esp32_camera_output_to_device(monkeypatch):
    camera = object()
    monkeypatch.setattr("thirdeye.network_camera.Esp32Camera", lambda *args, **kwargs: camera)
    app = object.__new__(ThirdEyeApp)
    app.settings = SimpleNamespace(
        camera_source="esp32", audio_source="pc", camera_host="0.0.0.0",
        camera_port=8081, camera_timeout=5, camera_token="test", voice_rate=0,
    )
    app._network_camera = None
    app._device_speaker = None
    app.cue_player = SimpleNamespace(set_output=lambda output: pytest.fail("ESP32 speaker selected"))

    assert app._uses_esp32_audio() is False
    assert isinstance(app._open_camera(), type(nullcontext()))
    assert app._network_camera is camera
    assert app._device_speaker is None


@pytest.mark.parametrize("transcribe_model", ["local", "gpt-4o-mini-transcribe"])
def test_pc_voice_control_uses_local_mic_with_esp32_camera(tmp_path, monkeypatch, transcribe_model):
    calls = []
    audio_path = tmp_path / "command.wav"
    audio_path.write_bytes(b"speech")

    class FakeListener:
        def __init__(self, *args):
            self.waits = 0

        def prepare(self):
            pass

        def close(self):
            pass

        def wait(self):
            self.waits += 1
            if self.waits == 2:
                raise KeyboardInterrupt

        def listen_text(self, **kwargs):
            if transcribe_model != "local":
                pytest.fail("Local transcription used in cloud mode")
            calls.append(("mic", None))
            calls.append(("transcribe", "local"))
            return "Find my bottle"

    class FakeRecorder:
        def __init__(self, device=None):
            calls.append(("mic", device))

        def record(self):
            return audio_path

    app = object.__new__(ThirdEyeApp)
    app.settings = SimpleNamespace(
        camera_source="esp32", audio_source="pc", wake_word="hi third eye",
        wake_language="en-US", wake_confidence=0.7, microphone_device=None,
        wake_model="small.en", transcribe_model=transcribe_model,
    )
    app.ai = SimpleNamespace(
        online=True,
        transcribe=lambda path: (calls.append(("transcribe", "cloud")) or "Find my bottle")
        if transcribe_model != "local" else pytest.fail("Cloud transcription used"),
        understand_command=lambda text: Command(Intent.FIND_OBJECT, "bottle"),
    )
    app.speaker = SimpleNamespace(
        wait=lambda: None,
        say=lambda text: calls.append(("speak", text)),
    )
    app._preload_realtime_navigation = lambda: None
    app.find_object = lambda target, listener: calls.append(("find", target))
    monkeypatch.setattr("thirdeye.app.WakeWordListener", FakeListener)
    monkeypatch.setattr("thirdeye.app.Recorder", FakeRecorder)
    monkeypatch.setattr("thirdeye.app.time.sleep", lambda seconds: None)
    monkeypatch.setattr("winsound.Beep", lambda *args: None)

    with pytest.raises(KeyboardInterrupt):
        app.run_voice_control()

    assert ("mic", None) in calls
    assert ("transcribe", "local" if transcribe_model == "local" else "cloud") in calls
    assert ("speak", "Hi.") in calls
    assert ("find", "bottle") in calls


@pytest.mark.parametrize("ending", ["done", None])
@pytest.mark.parametrize("backend", ["local", "gpt-live-transcribe"])
def test_pc_voice_question_keeps_original_words(tmp_path, monkeypatch, ending, backend):
    calls = []
    audio_path = tmp_path / "question.wav"
    audio_path.write_bytes(b"speech")
    transcripts = iter(["What does this label say?", ending])
    timeouts = []

    class FakeListener:
        def __init__(self, *args):
            self.waits = 0

        def prepare(self):
            pass

        def close(self):
            pass

        def wait(self):
            self.waits += 1
            if self.waits == 2:
                raise KeyboardInterrupt

        def listen_text(self, wait_timeout=None):
            timeouts.append(wait_timeout)
            return next(transcripts)

    class FakeRecorder:
        def __init__(self, device=None):
            pass

        def record(self, wait_timeout=None):
            return None if wait_timeout is not None else audio_path

    app = object.__new__(ThirdEyeApp)
    app.settings = SimpleNamespace(
        camera_source="esp32", audio_source="pc", wake_word="hi third eye",
        wake_language="en-US", wake_confidence=0.7, microphone_device=None,
        wake_model="small.en", transcribe_model=backend, openai_api_key="test-key",
    )
    app.ai = SimpleNamespace(
        online=True,
        transcribe=lambda path: pytest.fail("Cloud transcription used"),
        understand_command=lambda text: Command(Intent.VISUAL_QUESTION),
    )
    app.speaker = SimpleNamespace(wait=lambda: None, say=lambda text, **kwargs: None)
    app._preload_realtime_navigation = lambda: None
    app.answer_visual_question = calls.append
    monkeypatch.setattr("thirdeye.app.RealtimeTranscriber", FakeListener)
    monkeypatch.setattr("thirdeye.app.WakeWordListener", FakeListener)
    monkeypatch.setattr("thirdeye.app.Recorder", FakeRecorder)
    monkeypatch.setattr("thirdeye.app.time.sleep", lambda seconds: None)
    monkeypatch.setattr("winsound.Beep", lambda *args: None)

    with pytest.raises(KeyboardInterrupt):
        app.run_voice_control()

    assert calls == ["What does this label say?"]
    assert timeouts == [None, 30.0]
    assert app._visual_context is None


def test_pc_followup_does_not_require_a_second_wake(tmp_path, monkeypatch):
    heard = []
    wake_calls = []
    transcripts = iter([
        "What is in my hand?", "", "What color is it?",
        "Is the washing done?", "Done!",
    ])
    clips = [tmp_path / "first.wav", tmp_path / "second.wav"]
    for clip in clips:
        clip.write_bytes(b"speech")

    class FakeListener:
        def __init__(self, *args):
            pass

        def prepare(self):
            pass

        def wait(self):
            wake_calls.append(True)
            if len(wake_calls) == 2:
                raise KeyboardInterrupt

        def listen_text(self, **kwargs):
            assert kwargs.get("wait_timeout") == (30.0 if heard else None)
            return next(transcripts)

    class FakeRecorder:
        def __init__(self, device=None):
            pass

        def record(self, wait_timeout=None):
            return clips.pop(0) if clips else None

    app = object.__new__(ThirdEyeApp)
    app.settings = SimpleNamespace(
        camera_source="esp32", audio_source="pc", wake_word="hi third eye",
        wake_language="en-US", wake_confidence=0.7, microphone_device=None,
        wake_model="small.en", transcribe_model="local",
    )
    app.ai = SimpleNamespace(
        online=True,
        transcribe=lambda path: pytest.fail("Cloud transcription used"),
        understand_command=lambda text: Command(Intent.VISUAL_QUESTION),
    )
    app.speaker = SimpleNamespace(wait=lambda: None, say=lambda text, **kwargs: None)
    app._preload_realtime_navigation = lambda: None
    app.answer_visual_question = heard.append
    monkeypatch.setattr("thirdeye.app.WakeWordListener", FakeListener)
    monkeypatch.setattr("thirdeye.app.Recorder", FakeRecorder)
    monkeypatch.setattr("thirdeye.app.time.sleep", lambda seconds: None)
    monkeypatch.setattr("winsound.Beep", lambda *args: None)

    with pytest.raises(KeyboardInterrupt):
        app.run_voice_control()

    assert heard == ["What is in my hand?", "What color is it?", "Is the washing done?"]
    assert len(wake_calls) == 2
    assert app._visual_context is None


def test_scene_overview_can_be_followed_by_visual_question(tmp_path, monkeypatch):
    calls = []
    clips = [tmp_path / "scene.wav", tmp_path / "followup.wav"]
    for clip in clips:
        clip.write_bytes(b"speech")

    class FakeListener:
        def __init__(self, *args):
            pass

        def prepare(self):
            pass

        def wait(self):
            calls.append("wake")
            if calls.count("wake") == 2:
                raise KeyboardInterrupt

        def listen_text(self, **kwargs):
            if not clips:
                return "done"
            path = clips.pop(0)
            return "Describe scene" if path.name == "scene.wav" else "What is on my left?"

    class FakeRecorder:
        def __init__(self, device=None):
            pass

        def record(self, wait_timeout=None):
            return clips.pop(0) if clips else None

    app = object.__new__(ThirdEyeApp)
    app.settings = SimpleNamespace(
        camera_source="esp32", audio_source="pc", wake_word="hi third eye",
        wake_language="en-US", wake_confidence=0.7, microphone_device=None,
        wake_model="small.en", transcribe_model="local",
    )
    app.ai = SimpleNamespace(
        online=True,
        transcribe=lambda path: pytest.fail("Cloud transcription used"),
        understand_command=lambda text: Command(
            Intent.DESCRIBE_SCENE if text == "Describe scene" else Intent.VISUAL_QUESTION
        ),
    )
    app.speaker = SimpleNamespace(wait=lambda: None, say=lambda text, **kwargs: None)
    app._preload_realtime_navigation = lambda: None
    app.describe_scene = lambda mode: calls.append("overview")
    app.answer_visual_question = lambda question: calls.append(question)
    monkeypatch.setattr("thirdeye.app.WakeWordListener", FakeListener)
    monkeypatch.setattr("thirdeye.app.Recorder", FakeRecorder)
    monkeypatch.setattr("thirdeye.app.time.sleep", lambda seconds: None)
    monkeypatch.setattr("winsound.Beep", lambda *args: None)

    with pytest.raises(KeyboardInterrupt):
        app.run_voice_control()

    assert calls == ["wake", "overview", "What is on my left?", "wake"]
