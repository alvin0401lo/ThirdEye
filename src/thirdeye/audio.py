from __future__ import annotations

import gc
import base64
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from queue import Empty, Queue


def _is_windows() -> bool:
    return sys.platform == "win32"


def _powershell_string(value: str) -> str:
    # Keep speech and paths out of PowerShell source, including smart quotes.
    encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
    return f"[System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String('{encoded}'))"


class SpeechHighPass:
    """Two-pole 80 Hz filter with state carried across microphone packets."""

    def __init__(self, sample_rate: int = 16000):
        import math
        self.alpha = math.exp(-2 * math.pi * 80 / sample_rate)
        self.previous_input = None
        self.previous_first = 0.0
        self.first = self.second = 0.0

    def process(self, samples):
        import numpy as np
        values = np.asarray(samples, dtype=np.int16).reshape(-1)
        output = np.empty(values.size, dtype=np.float32)
        if values.size and self.previous_input is None:
            self.previous_input = float(values[0])
        for i, sample in enumerate(values):
            value = float(sample)
            self.first = self.alpha * (self.first + value - self.previous_input)
            self.second = self.alpha * (self.second + self.first - self.previous_first)
            self.previous_input = value
            self.previous_first = self.first
            output[i] = self.second
        return np.clip(output, -32768, 32767).astype(np.int16)


def _prepare_speech_samples(samples, sample_rate: int):
    import numpy as np

    mono = np.asarray(samples, dtype=np.int16).reshape(-1)
    if mono.size == 0:
        return mono

    frame_size = max(1, int(sample_rate * 0.02))
    usable = mono[: mono.size - (mono.size % frame_size)]
    if usable.size == 0:
        return mono
    frames = usable.astype(np.float32).reshape(-1, frame_size)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    noise_floor = float(np.percentile(rms, 20))
    active = np.flatnonzero(rms >= max(80.0, noise_floor * 3.0))

    if active.size:
        padding = int(sample_rate * 0.25)
        start = max(0, int(active[0]) * frame_size - padding)
        end = min(mono.size, (int(active[-1]) + 1) * frame_size + padding)
        mono = mono[start:end]

    peak = int(np.max(np.abs(mono.astype(np.int32)), initial=0))
    if 0 < peak < 24_000:
        gain = min(4.0, 28_000 / peak)
        mono = np.clip(mono.astype(np.float32) * gain, -32_768, 32_767).astype(np.int16)
    return mono


def _resample_audio(samples, source_rate: int, target_rate: int):
    import numpy as np

    if source_rate == target_rate or len(samples) == 0:
        return samples
    target_length = max(1, round(len(samples) * target_rate / source_rate))
    source_positions = np.arange(len(samples), dtype=np.float64)
    target_positions = np.linspace(0, len(samples) - 1, target_length)
    if samples.ndim == 1:
        return np.interp(target_positions, source_positions, samples).astype(samples.dtype)
    channels = [
        np.interp(target_positions, source_positions, samples[:, channel])
        for channel in range(samples.shape[1])
    ]
    return np.column_stack(channels).astype(samples.dtype)


def _phrase_match(
    heard: str,
    expected: str,
    minimum_confidence: float,
    *,
    allow_wake_partial: bool = False,
) -> tuple[bool, str, float]:
    normalized_words = "".join(
        character if character.isalnum() or character.isspace() else " "
        for character in heard.lower().replace("3rd", "third")
    ).split()
    expected_words = " ".join(expected.lower().split()).split()
    normalized_text = " ".join(normalized_words)
    expected_text = " ".join(expected_words)
    similarity = SequenceMatcher(None, normalized_text, expected_text).ratio()
    exact_match = (
        expected_words[0] in normalized_words
        if len(expected_words) == 1
        else expected_text in normalized_text
    )
    partial_match = False
    if allow_wake_partial and len(expected_words) >= 3 and similarity >= minimum_confidence:
        # Whisper occasionally drops the first or final word of a short wake phrase.
        partial_match = any(
            " ".join(expected_words[index:index + 2]) in normalized_text
            for index in range(len(expected_words) - 1)
        )
    return exact_match or partial_match, normalized_text, similarity


class IncompleteSpeechError(RuntimeError):
    pass


def _capture_utterance(
    device: int | str | None,
    sample_rate: int = 16_000,
    duration: float = 4.0,
    stop_event: threading.Event | None = None,
    wait_timeout: float | None = None,
    on_partial=None,
    require_complete: bool = False,
    on_chunk=None,
    end_silence_seconds: float = 0.45,
):
    import numpy as np
    import sounddevice as sd

    capture_rate = round(sd.query_devices(device, "input")["default_samplerate"])
    chunk_seconds = 0.05
    chunk_frames = max(1, round(capture_rate * chunk_seconds))
    silence_frames_required = round(capture_rate * end_silence_seconds)
    minimum_voice_frames = round(capture_rate * 0.20)
    wait_deadline = time.monotonic() + wait_timeout if wait_timeout is not None else None
    while stop_event is None or not stop_event.is_set():
        print(f"Speech listening: waiting for speech, up to {duration:g} seconds per utterance...", flush=True)
        with sd.InputStream(
            samplerate=capture_rate,
            channels=1,
            dtype="int16",
            device=device,
        ) as stream:
            pre_roll = deque(maxlen=12)
            chunks = []
            overflowed = False
            speech_started = False
            speech_deadline = 0.0
            voiced_frames = 0
            silent_frames = 0
            speech_frames = 0
            next_partial_frames = round(capture_rate * 0.8)
            while True:
                if stop_event is not None and stop_event.is_set():
                    return None
                samples, chunk_overflowed = stream.read(chunk_frames)
                overflowed = overflowed or chunk_overflowed
                if chunk_overflowed and require_complete:
                    raise IncompleteSpeechError("Microphone overflow: incomplete speech discarded; please repeat.")
                mono_chunk = samples.reshape(-1)
                peak = int(abs(mono_chunk.astype("int32")).max(initial=0))
                rms = float((mono_chunk.astype("float32") ** 2).mean() ** 0.5)
                active = peak >= 500 and rms >= 80.0

                if not speech_started:
                    if wait_deadline is not None and time.monotonic() >= wait_deadline:
                        return None
                    pre_roll.append(samples)
                    if not active:
                        continue
                    speech_started = True
                    speech_deadline = time.monotonic() + duration
                    chunks.extend(pre_roll)
                    if on_chunk is not None:
                        on_chunk(np.concatenate(chunks).reshape(-1), capture_rate)
                    voiced_frames += len(samples)
                    speech_frames = len(samples)
                    continue

                chunks.append(samples)
                if on_chunk is not None:
                    on_chunk(mono_chunk, capture_rate)
                speech_frames += len(samples)
                if active:
                    voiced_frames += len(samples)
                    silent_frames = 0
                else:
                    silent_frames += len(samples)
                if voiced_frames >= minimum_voice_frames and silent_frames >= silence_frames_required:
                    break
                if time.monotonic() >= speech_deadline:
                    if require_complete:
                        raise IncompleteSpeechError("Speech time limit reached: incomplete command discarded; please repeat a shorter sentence.")
                    break
                if on_partial is not None and voiced_frames >= minimum_voice_frames and speech_frames >= next_partial_frames:
                    on_partial(np.concatenate(chunks).reshape(-1), capture_rate)
                    next_partial_frames = speech_frames + round(capture_rate * 0.8)
        samples = np.concatenate(chunks) if chunks else np.empty((0, 1), dtype=np.int16)
        if overflowed:
            print("Microphone input overflowed; some audio was dropped.", flush=True)
        mono = samples.reshape(-1)
        peak = int(abs(mono.astype("int32")).max(initial=0))
        rms = float((mono.astype("float32") ** 2).mean() ** 0.5)
        print(f"Speech window: peak={peak}, rms={rms:.1f}", flush=True)
        if not speech_started or peak < 500 or rms < 80.0:
            print("No clear voice in window; listening again...", flush=True)
            continue
        prepared = _prepare_speech_samples(mono, capture_rate)
        return _resample_audio(prepared, capture_rate, sample_rate)
    return None


class WakeWordListener:
    def __init__(
        self,
        phrase: str = "hi third eye",
        language: str = "en-US",
        minimum_confidence: float = 0.70,
        device: int | str | None = None,
        model_name: str = "small.en",
    ) -> None:
        self.phrase = phrase
        self.language = language
        self.minimum_confidence = max(0.0, min(1.0, minimum_confidence))
        self.device = device
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            print(f"Loading local wake model: {self.model_name}", flush=True)
            self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        return self._model

    def unload_model(self) -> None:
        self._model = None
        gc.collect()

    def prepare(self) -> None:
        self._load_model()

    def _record_clip(
        self,
        duration: float = 4.0,
        stop_event: threading.Event | None = None,
    ) -> Path | None:
        samples = _capture_utterance(self.device, duration=duration, stop_event=stop_event)
        if samples is None:
            return None
        peak = int(abs(samples.astype("int32")).max(initial=0))
        rms = float((samples.astype("float32") ** 2).mean() ** 0.5)
        print(
            f"Speech clip: duration={len(samples) / 16_000:.2f}s, peak={peak}, rms={rms:.1f}",
            flush=True,
        )
        with tempfile.NamedTemporaryFile(prefix="thirdeye_wake_", suffix=".wav", delete=False) as temporary:
            path = Path(temporary.name)
        try:
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(samples.tobytes())
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return path

    def wait(self) -> str:
        result = self.wait_for(self.phrase)
        assert result is not None
        return result

    def capture_clip(
        self,
        window_seconds: float,
        stop_event: threading.Event | None = None,
    ) -> Path | None:
        return self._record_clip(window_seconds, stop_event)

    def matches_clip(
        self,
        path: Path,
        phrase: str,
        minimum_confidence: float | None = None,
        *,
        allow_wake_partial: bool = False,
    ) -> bool:
        expected = " ".join(phrase.lower().split())
        required_confidence = (
            self.minimum_confidence
            if minimum_confidence is None
            else max(0.0, min(1.0, minimum_confidence))
        )
        heard = self.transcribe_clip(path, hotwords=expected)
        matched, normalized_text, similarity = _phrase_match(
            heard,
            expected,
            required_confidence,
            allow_wake_partial=allow_wake_partial,
        )
        print(f"Speech model heard (expected '{expected}'): {heard or '[no speech]'}", flush=True)
        print(
            f"Phrase match: normalized={normalized_text or '[empty]'}, "
            f"score={similarity:.2f}, required={required_confidence:.2f}",
            flush=True,
        )
        return matched

    def transcribe_clip(self, path: Path, hotwords: str | None = None) -> str:
        return self._transcribe(str(path), hotwords)

    def transcribe_samples(self, samples, hotwords: str | None = None) -> str:
        import numpy as np

        # faster-whisper accepts normalized mono audio at 16 kHz directly.
        return self._transcribe(np.asarray(samples, dtype=np.float32) / 32768.0, hotwords)

    def _transcribe(self, source, hotwords: str | None = None) -> str:
        segments, _ = self._load_model().transcribe(
            source,
            beam_size=1,
            language=self.language.split("-", 1)[0].lower(),
            condition_on_previous_text=False,
            vad_filter=False,
            hotwords=hotwords,
            no_speech_threshold=None,
            log_prob_threshold=None,
            without_timestamps=True,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    def listen_text(
        self,
        duration: float = 12.0,
        stop_event: threading.Event | None = None,
        wait_timeout: float | None = None,
        hotwords: str | None = None,
        preview: bool = True,
        end_silence_seconds: float = 0.45,
    ) -> str | None:
        self.prepare()
        if not preview:
            samples = _capture_utterance(
                self.device, duration=duration, stop_event=stop_event,
                wait_timeout=wait_timeout, require_complete=True,
                end_silence_seconds=end_silence_seconds,
            )
            if samples is None or (stop_event is not None and stop_event.is_set()):
                return None
            started = time.monotonic()
            text = self.transcribe_samples(samples, hotwords)
            if stop_event is not None and stop_event.is_set():
                return None
            print(f"[ASR FINAL] {text or '[no speech]'}", flush=True)
            print(f"Wake ASR end-to-final: {time.monotonic() - started:.2f}s", flush=True)
            return text
        finished = threading.Event()
        future = None
        last_partial = ""

        def preview(samples, rate):
            nonlocal last_partial
            try:
                prepared = _resample_audio(_prepare_speech_samples(samples, rate), rate, 16_000)
                text = self.transcribe_samples(prepared, hotwords)
                if not finished.is_set() and not (stop_event is not None and stop_event.is_set()):
                    if text and text != last_partial:
                        print(f"[ASR PARTIAL] {text}", flush=True)
                        last_partial = text
            except Exception as exc:
                if not finished.is_set():
                    print(f"ASR preview unavailable: {exc}", flush=True)

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="thirdeye-asr") as worker:
            def submit_preview(samples, rate):
                nonlocal future
                # Skip previews while busy: never accumulate outdated inference jobs.
                if future is None or future.done():
                    future = worker.submit(preview, samples.copy(), rate)

            try:
                samples = _capture_utterance(
                    self.device, duration=duration, stop_event=stop_event,
                    wait_timeout=wait_timeout, on_partial=submit_preview,
                    require_complete=True,
                )
            finally:
                finished.set()
            if samples is None or (stop_event is not None and stop_event.is_set()):
                return None
            started = time.monotonic()
            # One worker serializes model access; final follows at most one preview.
            def finalize():
                if stop_event is not None and stop_event.is_set():
                    return None
                return self.transcribe_samples(samples, hotwords)

            text = worker.submit(finalize).result()
            if text is None or (stop_event is not None and stop_event.is_set()):
                return None
            print(f"[ASR FINAL] {text or '[no speech]'}", flush=True)
            print(f"ASR end-to-final: {time.monotonic() - started:.2f}s", flush=True)
            return text

    def wait_for(
        self,
        phrase: str,
        stop_event: threading.Event | None = None,
        window_seconds: float = 4.0,
        minimum_confidence: float | None = None,
    ) -> str | None:
        expected = " ".join(phrase.lower().split())
        while stop_event is None or not stop_event.is_set():
            try:
                heard = self.listen_text(
                    window_seconds, stop_event, hotwords=expected, preview=False,
                    end_silence_seconds=0.25 if expected == "done" else 0.45,
                )
            except IncompleteSpeechError as exc:
                print(f"Speech input retry: {exc}", flush=True)
                continue
            if heard is None:
                return None
            confidence = self.minimum_confidence if minimum_confidence is None else minimum_confidence
            matched, normalized, score = _phrase_match(heard, expected, confidence, allow_wake_partial=True)
            print(f"Phrase match: normalized={normalized or '[empty]'}, score={score:.2f}, required={confidence:.2f}", flush=True)
            if matched:
                return expected
            print("Phrase not matched; listening again...", flush=True)
        return None


class Recorder:
    def __init__(
        self,
        sample_rate: int = 16_000,
        seconds: float = 5.0,
        device: int | str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.seconds = seconds
        self.device = device

    def record(self, wait_timeout: float | None = None) -> Path | None:
        import sounddevice as sd

        microphone = sd.query_devices(self.device, "input")
        print(f"Microphone: {microphone['name']}")
        samples = _capture_utterance(
            self.device,
            sample_rate=self.sample_rate,
            duration=self.seconds,
            wait_timeout=wait_timeout,
        )
        if samples is None:
            if wait_timeout is not None:
                return None
            raise RuntimeError("Speech recording stopped")
        print(f"Speech audio: {len(samples) / self.sample_rate:.1f} seconds")
        output = Path(tempfile.gettempdir()) / "thirdeye_command.wav"
        with wave.open(str(output), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            handle.writeframes(samples.tobytes())
        return output


class Speaker:
    def __init__(self, rate_adjustment: int = 0, device: int | str | None = None) -> None:
        self._rate = max(-10, min(10, rate_adjustment))
        self._device = device
        self._queue: Queue[str | None] = Queue(maxsize=1)
        self._closed = False
        self._process: subprocess.Popen[bytes] | None = None
        self._interrupted_processes: set[int] = set()
        self._process_lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, name="thirdeye-tts", daemon=True)
        self._worker.start()

    def say(self, text: str, interrupt: bool = False) -> None:
        if not text.strip() or self._closed:
            return
        if interrupt:
            with self._process_lock:
                if self._process is not None and self._process.poll() is None:
                    self._interrupted_processes.add(id(self._process))
                    self._process.terminate()
            if self._device is not None:
                import sounddevice as sd

                sd.stop()
        try:
            self._queue.get_nowait()
            self._queue.task_done()
        except Empty:
            pass
        self._queue.put_nowait(text)

    def close(self) -> None:
        if self._closed:
            return
        self._queue.join()
        self._closed = True
        self._queue.put(None)
        self._worker.join(timeout=5)

    def wait(self) -> None:
        self._queue.join()

    def _run(self) -> None:
        while True:
            text = self._queue.get()
            if text is None:
                self._queue.task_done()
                return
            try:
                self._speak(text)
            except Exception as exc:
                print(f"Speech output failed: {exc}")
            finally:
                self._queue.task_done()

    def _speak(self, text: str) -> None:
        wave_path = Path(tempfile.gettempdir()) / "thirdeye_speech.wav"
        if _is_windows() and self._device is not None:
            speech = _powershell_string(text)
            path_expression = _powershell_string(str(wave_path))
            script = (
                "$speaker=New-Object -ComObject SAPI.SpVoice; "
                "$stream=New-Object -ComObject SAPI.SpFileStream; "
                f"$stream.Open({path_expression},3,$false); "
                "$speaker.AudioOutputStream=$stream; "
                f"$speaker.Rate={self._rate}; [void]$speaker.Speak({speech}); "
                "$stream.Close()"
            )
            command = ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]
        elif _is_windows():
            speech = _powershell_string(text)
            script = (
                "Add-Type -AssemblyName System.Speech; "
                "$speaker=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$speaker.Rate={self._rate}; $speaker.Speak({speech}); $speaker.Dispose()"
            )
            command = ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]
        else:
            executable = shutil.which("espeak-ng")
            if executable is None:
                raise RuntimeError("Install eSpeak NG for Linux speech output: sudo apt install espeak-ng")
            words_per_minute = max(80, min(450, 165 + self._rate * 8))
            command = [executable, "-v", "en-us", "-s", str(words_per_minute), "-w", str(wave_path), text]
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        with self._process_lock:
            self._process = process
        _, error = process.communicate()
        with self._process_lock:
            interrupted = id(process) in self._interrupted_processes
            self._interrupted_processes.discard(id(process))
            if self._process is process:
                self._process = None
        if process.returncode != 0 and not interrupted:
            raise RuntimeError(error.decode(errors="replace").strip() or "Windows speech synthesis failed")
        if not _is_windows() or self._device is not None:
            import numpy as np
            import sounddevice as sd

            with wave.open(str(wave_path), "rb") as handle:
                samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
                channels = handle.getnchannels()
                sample_rate = handle.getframerate()
            if channels > 1:
                samples = samples.reshape(-1, channels)
            output_rate = round(sd.query_devices(self._device, "output")["default_samplerate"])
            samples = _resample_audio(samples, sample_rate, output_rate)
            try:
                sd.play(samples, output_rate, device=self._device)
                sd.wait()
            finally:
                wave_path.unlink(missing_ok=True)
