from __future__ import annotations

import queue
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

from thirdeye.audio import _powershell_string, _resample_audio


CUE_FILES = {
    "left": "left.wav",
    "right": "right.wav",
    "up": "up.wav",
    "down": "down.wav",
    "forward": "forward.wav",
    "hold": "hold.wav",
    "show your hand": "show_hand.wav",
    "object not found": "object_not_found.wav",
    "object detected": "detected.wav",
    "locking target": "detected.wav",
    "object centered": "ok.wav",
    "move camera left": "camera_left.wav",
    "move camera right": "camera_right.wav",
    "move camera up": "camera_up.wav",
    "move camera down": "camera_down.wav",
}

BEEP_CUES = {
    "left": (660, 90),
    "right": (880, 90),
    "up": (1040, 90),
    "down": (520, 90),
    "forward": (760, 160),
    "hold": (620, 120),
    "show your hand": (440, 180),
    "object not found": (360, 180),
    "object detected": (980, 140),
    "locking target": (980, 140),
    "object centered": (1200, 110),
    "move camera left": (660, 90),
    "move camera right": (880, 90),
    "move camera up": (1040, 90),
    "move camera down": (520, 90),
}


class CuePlayer:
    def __init__(self, audio_dir: str | Path, output=None) -> None:
        self._audio_dir = Path(audio_dir)
        self._output = output
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=1)
        self._closed = False
        self._worker = threading.Thread(target=self._run, name="thirdeye-cue-audio", daemon=True)
        self._worker.start()

    def play(self, cue: str) -> None:
        key = _cue_key(cue)
        if not key or self._closed:
            return
        try:
            self._queue.get_nowait()
            self._queue.task_done()
        except queue.Empty:
            pass
        self._queue.put_nowait(key)

    def wait(self) -> None:
        self._queue.join()

    def set_output(self, output) -> None:
        self._output = output

    def close(self) -> None:
        if self._closed:
            return
        self._queue.join()
        self._closed = True
        self._queue.put(None)
        self._worker.join(timeout=3)

    def _run(self) -> None:
        while True:
            cue = self._queue.get()
            if cue is None:
                self._queue.task_done()
                return
            try:
                self._play_cue(cue)
            except Exception as exc:
                print(f"Cue audio failed: {exc}", flush=True)
            finally:
                self._queue.task_done()

    def _play_cue(self, cue: str) -> None:
        import winsound

        path = self._audio_dir / CUE_FILES.get(cue, f"{cue.replace(' ', '_')}.wav")
        if self._output is not None:
            if path.exists():
                self._output.play_wav(path)
            else:
                print(f"ESP32 cue unavailable: {path.name}", flush=True)
            return
        if path.exists():
            winsound.PlaySound(str(path), winsound.SND_FILENAME)
            return
        frequency, duration_ms = BEEP_CUES.get(cue, (700, 100))
        winsound.Beep(frequency, duration_ms)


def _cue_key(cue: str) -> str:
    return " ".join(cue.strip().lower().split())


class Esp32AudioOutput:
    """Serve mono 16 kHz WAV to the device's independent HTTP speaker."""

    def __init__(self, device, rate: int = 0) -> None:
        self._device = device
        self._rate = max(-10, min(10, rate))
        self._lock = threading.Lock()
        self._speech_cache: dict[str, bytes] = {}

    def _load_wav_pcm(self, path: str | Path) -> bytes:
        import numpy as np

        with wave.open(str(path), "rb") as handle:
            if handle.getsampwidth() != 2:
                raise ValueError(f"Expected 16-bit PCM WAV: {path}")
            samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
            if handle.getnchannels() > 1:
                samples = samples.reshape(-1, handle.getnchannels()).astype(np.int32).mean(axis=1).astype(np.int16)
            samples = _resample_audio(samples, handle.getframerate(), 16_000)
        # Remove only leading/trailing TTS silence, not pauses inside sentences.
        active = np.flatnonzero(np.abs(samples.astype(np.int32)) >= 80)
        if active.size:
            samples = samples[max(0, active[0] - 640):min(len(samples), active[-1] + 641)]
        peak = int(np.max(np.abs(samples.astype(np.int32)), initial=0))
        if peak >= 256:
            gain = min(4.0, 30_000 / peak)
            samples = np.clip(samples.astype(np.float32) * gain, -30_000, 30_000).astype(np.int16)
        return samples.tobytes()

    def play_wav(self, path: str | Path) -> None:
        self._play_pcm(self._load_wav_pcm(path))

    def say(self, text: str) -> None:
        if not text.strip():
            return
        if text in self._speech_cache:
            self._play_pcm(self._speech_cache[text])
            return
        with tempfile.NamedTemporaryFile(prefix="thirdeye_speech_", suffix=".wav", delete=False) as handle:
            path = Path(handle.name)
        speech = _powershell_string(text)
        path_expression = _powershell_string(str(path))
        script = (
            "$ErrorActionPreference='Stop'; Add-Type -AssemblyName System.Speech; "
            "$speaker=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$format=[System.Speech.AudioFormat.SpeechAudioFormatInfo]::new(16000, "
            "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
            "[System.Speech.AudioFormat.AudioChannel]::Mono); "
            f"$speaker.Rate={self._rate}; "
            f"try {{ $speaker.SetOutputToWaveFile({path_expression}, $format); "
            f"[void]$speaker.Speak({speech}) }} finally {{ $speaker.Dispose() }}"
        )
        try:
            started_at = time.monotonic()
            try:
                subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                    check=True, capture_output=True, timeout=60,
                )
            except subprocess.CalledProcessError as exc:
                detail = (exc.stderr or b"").decode(errors="replace").strip()
                raise RuntimeError(f"Windows WAV synthesis failed: {detail or 'no error details'}") from exc
            print(f"Speech WAV synthesis: {time.monotonic() - started_at:.2f}s", flush=True)
            pcm = self._load_wav_pcm(path)
            # Bound the cache; frequent short responses avoid another PowerShell launch.
            if len(text) <= 80:
                if len(self._speech_cache) >= 32:
                    self._speech_cache.pop(next(iter(self._speech_cache)))
                self._speech_cache[text] = pcm
            self._play_pcm(pcm)
        finally:
            path.unlink(missing_ok=True)

    def _play_pcm(self, pcm: bytes) -> None:
        with self._lock:
            try:
                self._device.play_speaker_pcm(pcm)
            finally:
                self._device.clear_audio_buffer()
