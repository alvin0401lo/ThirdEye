import io
import socket
import threading
import time
import wave
from contextlib import nullcontext
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from thirdeye.app import ThirdEyeApp
from thirdeye.camera import CameraError
from thirdeye.cue_audio import Esp32AudioOutput
from thirdeye.network_camera import Esp32Camera


class FakeDevice:
    audio_connected = False

    def __init__(self):
        self.pcm = None
        self.cleared = False

    def play_speaker_pcm(self, pcm):
        self.pcm = pcm

    def clear_audio_buffer(self):
        self.cleared = True


def test_wav_resampled_and_amplified(tmp_path):
    path = tmp_path / "cue.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(np.full((2205, 2), 1000, dtype="<i2").tobytes())
    device = FakeDevice()
    Esp32AudioOutput(device).play_wav(path)
    assert len(device.pcm) == 3200
    assert np.frombuffer(device.pcm, dtype="<i2").max() == 4000
    assert device.cleared


def test_gain_does_not_clip(tmp_path):
    path = tmp_path / "loud.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.array([-20000, 0, 20000], dtype="<i2").tobytes())
    device = FakeDevice()
    Esp32AudioOutput(device).play_wav(path)
    assert np.frombuffer(device.pcm, dtype="<i2").tolist() == [-30000, 0, 30000]
    assert device.cleared


def test_wav_trims_only_outer_silence(tmp_path):
    path = tmp_path / "padded.wav"
    speech = np.concatenate([np.full(1600, 1000), np.zeros(1600), np.full(1600, 1000)]).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.concatenate([np.zeros(8000, dtype="<i2"), speech, np.zeros(8000, dtype="<i2")]).tobytes())
    device = FakeDevice()
    Esp32AudioOutput(device).play_wav(path)
    played = np.frombuffer(device.pcm, dtype="<i2")
    assert len(played) == len(speech) + 1280
    assert not played[2240:3840].any()  # The pause inside the utterance remains.


def test_short_tts_cached_after_first_generation(monkeypatch):
    generated = []
    def synthesize(args, **kwargs):
        import base64
        import re
        encoded_path = re.findall(r"FromBase64String\('([^']+)'", args[-1])[0]
        path = base64.b64decode(encoded_path).decode("utf-8")
        with wave.open(path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\xe8\x03" * 320)
        generated.append(path)
    monkeypatch.setattr("thirdeye.cue_audio.subprocess.run", synthesize)
    device = FakeDevice()
    output = Esp32AudioOutput(device)
    output.say("Ready.")
    first = device.pcm
    output.say("Ready.")
    assert len(generated) == 1 and device.pcm == first


def test_failed_http_playback_clears_microphone_buffer():
    device = FakeDevice()
    def fail(pcm):
        raise CameraError("HTTP playback failed")
    device.play_speaker_pcm = fail
    with pytest.raises(CameraError, match="HTTP playback failed"):
        Esp32AudioOutput(device)._play_pcm(b"\x00\x00")
    assert device.cleared


def test_tts_encodes_unicode_quotes_and_command_characters(monkeypatch):
    import base64
    import re
    generated = []
    text = "A person's arm; $(Get-Process) \u2018quoted\u2019 \u201cwords\u201d\nnext line"
    def synthesize(args, **kwargs):
        script = args[-1]
        encoded = re.findall(r"FromBase64String\('([^']+)'", script)
        path, decoded = [base64.b64decode(value).decode("utf-8") for value in encoded]
        assert decoded == text and text not in script
        assert "$ErrorActionPreference='Stop'" in script and "finally" in script
        with wave.open(path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\xe8\x03" * 320)
        generated.append(path)
    monkeypatch.setattr("thirdeye.cue_audio.subprocess.run", synthesize)
    device = FakeDevice()
    Esp32AudioOutput(device).say(text)
    assert generated and device.pcm
    from pathlib import Path
    assert not Path(generated[0]).exists()


def test_tts_reports_actual_error_and_cleans_temp_file(monkeypatch):
    import base64
    import re
    import subprocess
    from pathlib import Path
    paths = []
    def fail(args, **kwargs):
        value = re.findall(r"FromBase64String\('([^']+)'", args[-1])[0]
        paths.append(Path(base64.b64decode(value).decode("utf-8")))
        raise subprocess.CalledProcessError(1, args, stderr=b"Speech voice unavailable")
    monkeypatch.setattr("thirdeye.cue_audio.subprocess.run", fail)
    device = FakeDevice()
    with pytest.raises(RuntimeError, match="Speech voice unavailable"):
        Esp32AudioOutput(device).say("Ready.")
    assert paths and not paths[0].exists() and device.pcm is None


@pytest.mark.parametrize("state", ["done", "error"])
def test_http_wav_without_microphone_and_authenticated_completion(state):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    camera = Esp32Camera(host="127.0.0.1", port=port, token="secret")
    base = f"http://127.0.0.1:{port}"
    results = []
    errors = []
    pcm = b"\x01\x00" * 1600

    def play():
        try:
            camera.play_speaker_pcm(pcm)
            results.append("done")
        except CameraError as exc:
            errors.append(str(exc))

    worker = threading.Thread(target=play)
    try:
        assert not camera.speaker_http_connected and not camera.audio_connected
        with pytest.raises(HTTPError) as exc:
            urlopen(base + "/stream.wav?token=wrong")
        assert exc.value.code == 403 and not camera.speaker_http_connected
        with urlopen(base + "/stream.wav?token=secret") as response:
            assert response.status == 204
        assert camera.speaker_http_connected
        worker.start()
        deadline = time.monotonic() + 3
        body = None
        while time.monotonic() < deadline:
            with urlopen(base + "/stream.wav?token=secret") as response:
                if response.status == 200:
                    body = response.read()
                    playback_id = response.headers["X-Playback-Id"]
                    assert int(response.headers["Content-Length"]) == len(body)
                    break
            time.sleep(0.01)
        assert body is not None and worker.is_alive()
        with wave.open(io.BytesIO(body)) as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
            assert wav.readframes(wav.getnframes()) == pcm
        with urlopen(base + "/stream.wav?token=secret") as response:
            assert response.status == 204  # A job is not replayed by a second poll.
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(base + "/speaker/status?token=secret&id=999&state=done", method="POST", data=b""))
        assert exc.value.code == 409 and worker.is_alive()
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(base + f"/speaker/status?token=wrong&id={playback_id}&state=done", method="POST", data=b""))
        assert exc.value.code == 403 and worker.is_alive()
        with urlopen(Request(base + f"/speaker/status?token=secret&id={playback_id}&state={state}", method="POST", data=b"")) as response:
            assert response.status == 204
        worker.join(2)
        assert not worker.is_alive()
        assert results == (["done"] if state == "done" else [])
        assert bool(errors) == (state == "error")
        assert camera._http_speaker_job is None
    finally:
        camera.release()
        if worker.ident is not None:
            worker.join(2)


def test_speaker_pcm_format_validation():
    camera = object.__new__(Esp32Camera)
    for pcm in (b"", b"\x00", bytes(1920002)):
        with pytest.raises(CameraError, match="PCM"):
            camera.play_speaker_pcm(pcm)


@pytest.mark.parametrize("state", ["done", "error"])
def test_download_retry_is_bounded_and_never_replays_a_finished_job(state):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    camera = Esp32Camera(host="127.0.0.1", port=port, token="secret")
    base = f"http://127.0.0.1:{port}"
    errors = []
    def play():
        try:
            camera.play_speaker_pcm(b"\x01\x00" * 1600)
        except CameraError as exc:
            errors.append(str(exc))
    worker = threading.Thread(target=play)
    try:
        worker.start()
        deadline = time.monotonic() + 2
        while camera._http_speaker_job is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert camera._http_speaker_job is not None
        url = base + "/stream.wav?token=secret&download=unique-board-request"
        bodies, ids = [], []
        for attempt in range(3):
            with urlopen(url, timeout=2) as response:
                assert response.status == 200
                bodies.append(response.read())
                ids.append(response.headers["X-Playback-Id"])
            # Normal polling or a different transaction cannot replay this job.
            for suffix in ("", "&download=other-request"):
                with urlopen(base + "/stream.wav?token=secret" + suffix, timeout=2) as response:
                    assert response.status == 204
        assert bodies[0] == bodies[1] == bodies[2] and len(set(ids)) == 1
        with pytest.raises(HTTPError) as exc:
            urlopen(url, timeout=2)
        assert exc.value.code == 409
        with pytest.raises(HTTPError) as exc:
            urlopen(url.replace("token=secret", "token=wrong"), timeout=2)
        assert exc.value.code == 403
        status_url = base + f"/speaker/status?token=secret&id={ids[0]}&state={state}&reason=i2s_write_failed"
        with urlopen(Request(status_url, method="POST", data=b""), timeout=2) as response:
            assert response.status == 204
        worker.join(2)
        assert not worker.is_alive()
        assert bool(errors) == (state == "error")
        if errors:
            assert "i2s_write_failed" in errors[0]
        with urlopen(url, timeout=2) as response:
            assert response.status == 204
    finally:
        camera.release()
        worker.join(2)




def test_speaker_diagnostic_does_not_require_microphone(tmp_path):
    calls = []
    app = object.__new__(ThirdEyeApp)
    app.settings = SimpleNamespace(cue_audio_dir=tmp_path)
    app._uses_esp32_audio = lambda: True
    app._open_camera = lambda: nullcontext()
    app._network_camera = SimpleNamespace(speaker_http_connected=True, audio_connected=False)
    app._device_speaker = SimpleNamespace(play_wav=lambda path: calls.append(path))
    app.test_esp32_speaker()
    assert calls == [tmp_path / "camera_down.wav"]


def test_speaker_diagnostic_does_not_claim_success_on_failure(tmp_path, capsys):
    app = object.__new__(ThirdEyeApp)
    app.settings = SimpleNamespace(cue_audio_dir=tmp_path)
    app._uses_esp32_audio = lambda: True
    app._open_camera = lambda: nullcontext()
    app._network_camera = SimpleNamespace(speaker_http_connected=True)
    def fail(path):
        raise CameraError("HTTP playback failed")
    app._device_speaker = SimpleNamespace(play_wav=fail)
    with pytest.raises(CameraError):
        app.test_esp32_speaker()
    assert "reported playback complete" not in capsys.readouterr().out


def test_firmware_transport_preserves_camera_and_pins():
    from pathlib import Path
    sketch = (Path(__file__).resolve().parent.parent / "firmware/thirdeye_ai_device/thirdeye_ai_device.ino").read_text(encoding="utf-8")
    for setting in ("#define CAMERA_FRAME_INTERVAL_MS 88", "config.jpeg_quality = hasPsram ? 10 : 12;",
                    "#define I2S_BCLK 2", "#define I2S_WS 1", "#define I2S_MIC_SD 3",
                    "#define I2S_SPK_DOUT 14", ".mck_io_num = I2S_PIN_NO_CHANGE"):
        assert setting in sketch
    assert "Audio firmware: HTTP_WAV_V1" in sketch
    assert "xTaskCreate(playHttpSpeaker" in sketch
    assert "speakerQueue" not in sketch and "Speaker START received" not in sketch
    assert '"/stream.wav?token="' in sketch and '"/speaker/status?token="' in sketch
