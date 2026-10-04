from __future__ import annotations

import argparse
import multiprocessing
import re
import sys
import tempfile
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path

from thirdeye.ai_service import AIService, AIUnavailable, is_english_command_text
from thirdeye.audio import IncompleteSpeechError, Recorder, Speaker, SpeechHighPass, WakeWordListener, _prepare_speech_samples
from thirdeye.camera import CameraError, Webcam
from thirdeye.camera_worker import CameraWorker
from thirdeye.config import Settings
from thirdeye.cue_audio import CuePlayer, Esp32AudioOutput
from thirdeye.models import Command, Intent, SceneMode
from thirdeye.repo_guidance_audio import RepoGuidanceAnnouncer
from thirdeye.repo_object_finding import RepoObjectFinder
from thirdeye.realtime_audio import RealtimeTranscriber
from thirdeye.scene import assess_frame, is_text_question, render_scene_analysis, select_best_frame


SCENE_CLASSES = [
    "person",
    "door",
    "stairs",
    "chair",
    "table",
    "car",
    "bicycle",
    "backpack",
    "box",
    "sign",
]


def _listen_for_done(
    language: str,
    device: int | str | None,
    model_name: str,
    stop_event,
    completed_event,
    start_event,
) -> None:
    listener = WakeWordListener("done", language, 1.0, device, model_name)
    try:
        listener.prepare()
        print("Done listener ready", flush=True)
        while not start_event.wait(0.1):
            if stop_event.is_set():
                return
        if listener.wait_for(
            "done",
            stop_event,
            window_seconds=1.5,
            minimum_confidence=1.0,
        ):
            completed_event.set()
    except Exception as exc:
        print(f"Completion voice input unavailable: {exc}", flush=True)


class ThirdEyeApp:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ai = AIService(
            settings.openai_api_key,
            settings.openai_model,
            settings.transcribe_model,
            settings.vision_model,
            settings.language,
        )
        self.speaker = Speaker(settings.voice_rate, settings.speaker_device)
        self.cue_player = CuePlayer(settings.cue_audio_dir)
        self._detector = None
        self._hand_tracker = None
        self._network_camera = None
        self._device_speaker = None
        self._camera_worker = None
        self._esp32_voice_thresholds = None
        self._visual_context = None

    def _open_camera(self):
        if self.settings.camera_source == "esp32":
            if self._network_camera is None:
                from thirdeye.network_camera import Esp32Camera

                self._network_camera = Esp32Camera(
                    self.settings.camera_host,
                    self.settings.camera_port,
                    self.settings.camera_timeout,
                    self.settings.camera_token,
                    companion_bridge_url=getattr(self.settings, "companion_bridge_url", None),
                    companion_room=getattr(self.settings, "companion_room", "THIRDEYE"),
                )
                if self._uses_esp32_audio():
                    self._device_speaker = Esp32AudioOutput(self._network_camera, self.settings.voice_rate)
                    self.cue_player.set_output(self._device_speaker)
            return nullcontext(self._network_camera)
        if self.settings.camera_source != "webcam":
            raise CameraError("THIRDEYE_CAMERA_SOURCE must be 'webcam' or 'esp32'")
        return Webcam(
            self.settings.camera_index,
            self.settings.camera_width,
            self.settings.camera_height,
        )

    def close(self) -> None:
        if self._camera_worker is not None:
            self._camera_worker.close()
        if self._network_camera is not None:
            self._network_camera.release()
        if self._hand_tracker is not None:
            self._hand_tracker.close()
        self.cue_player.close()
        self.speaker.close()

    def _start_camera_worker(self) -> CameraWorker:
        if self._camera_worker is None:
            self._camera_worker = CameraWorker(
                self._open_camera,
                self.settings.camera_source,
                mirror=self.settings.mirror_hand_view,
            )
            self._camera_worker.start()
        return self._camera_worker

    @contextmanager
    def _video_session(self, profile: str = "live"):
        if getattr(self.settings, "camera_source", None) == "esp32":
            if self._network_camera is None:
                self._open_camera()
            with self._network_camera.video_session(profile=profile):
                worker = getattr(self, "_camera_worker", None)
                if worker is not None:
                    worker.clear_frames()
                yield
        else:
            yield

    def _capture_scene_frame(self, count: int = 3):
        with self._video_session():
            return self._select_scene_frame(count)

    def _select_scene_frame(self, count: int):
        worker = self._start_camera_worker()
        previous = worker.read(timeout=self.settings.camera_timeout)
        deadline = time.monotonic() + self.settings.camera_timeout
        packets = []
        for _ in range(count):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                packet = worker.read(previous.frame_id, timeout=remaining)
            except CameraError:
                break
            previous = packet
            packets.append(packet)
        now = time.monotonic()
        frames = [packet.image for packet in packets if now - packet.timestamp <= 1.5]
        if not frames:
            raise CameraError("No fresh camera frame available. Please check the camera connection.")
        return select_best_frame(frames)

    def _capture_text_frame(self):
        if getattr(getattr(self, "settings", None), "camera_source", None) != "esp32":
            return self._capture_scene_frame(count=5)
        with self._open_camera():
            frame = self._network_camera.capture_still(timeout=max(8.0, self.settings.camera_timeout))
            if self.settings.mirror_hand_view:
                import cv2

                frame = cv2.flip(frame, 1)
            print(f"Text photo: {frame.shape[1]}x{frame.shape[0]}", flush=True)
            return frame, assess_frame(frame)

    def _preload_realtime_navigation(self) -> None:
        print("Preloading navigation models and camera...")
        self._load_detector()
        self._load_hand_tracker()
        camera_worker = self._start_camera_worker()
        if self.settings.camera_source == "esp32":
            print("ESP32 video on demand; waiting for a visual task or website viewer.")
            return
        try:
            packet = camera_worker.read(timeout=self.settings.camera_timeout)
            height, width = packet.image.shape[:2]
            print(f"Navigation camera ready: {width}x{height}")
        except CameraError as exc:
            print(f"Navigation camera is not ready yet: {exc}")

    def _uses_esp32_audio(self) -> bool:
        source = getattr(self.settings, "audio_source", "auto")
        if source not in {"auto", "pc", "esp32"}:
            raise ValueError("THIRDEYE_AUDIO_SOURCE must be 'auto', 'pc', or 'esp32'")
        return source == "esp32" or (source == "auto" and self.settings.camera_source == "esp32")

    def _calibrate_esp32_microphone(self, stop_event: threading.Event | None = None) -> bool:
        import numpy as np

        sample_rate = 16_000
        target_frames = int(sample_rate * 0.8)
        high_pass = SpeechHighPass(sample_rate)
        received_frames = 0
        peaks = []
        levels = []
        deadline = time.monotonic() + 4.0
        print("Calibrating ESP32 microphone; stay quiet for one second...", flush=True)
        while received_frames < target_frames and time.monotonic() < deadline:
            if stop_event is not None and stop_event.is_set():
                return False
            try:
                chunk = self._network_camera.read_audio(timeout=0.25)
            except CameraError:
                continue
            samples = high_pass.process(np.frombuffer(chunk[: len(chunk) & ~1], dtype="<i2")).astype(np.int32)
            if samples.size == 0:
                continue
            received_frames += samples.size
            peaks.append(int(np.abs(samples).max()))
            levels.append(float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))))
        if received_frames < target_frames:
            raise CameraError(
                f"ESP32 microphone calibration needs 0.8s of audio; received "
                f"{received_frames / sample_rate:.1f}s in 4s"
            )
        noise_peak = float(np.percentile(peaks, 20))
        noise_rms = float(np.percentile(levels, 20))
        if noise_rms >= 1000.0:
            raise CameraError(
                f"ESP32 microphone calibration too loud (noise_rms={noise_rms:.0f}); "
                "stay quiet and retry"
            )
        self._esp32_voice_thresholds = (max(500.0, noise_peak * 2.5), max(80.0, noise_rms * 3.0))
        print(
            f"ESP32 microphone ready (80 Hz high-pass): noise_rms={noise_rms:.1f}, "
            f"voice_peak>={self._esp32_voice_thresholds[0]:.0f}, "
            f"voice_rms>={self._esp32_voice_thresholds[1]:.0f}",
            flush=True,
        )
        return True

    def _record_esp32_audio(
        self,
        seconds: float,
        stop_event: threading.Event | None = None,
        wait_timeout: float | None = None,
        *,
        on_chunk=None,
        require_complete: bool = False,
        return_samples: bool = False,
        min_voiced_seconds: float = 0.20,
    ):
        if self._network_camera is None:
            raise CameraError("ESP32 microphone server is not running")
        import numpy as np

        if self._esp32_voice_thresholds is None:
            if not self._calibrate_esp32_microphone(stop_event):
                return None
        voice_peak_threshold, voice_rms_threshold = self._esp32_voice_thresholds
        wait_deadline = time.monotonic() + wait_timeout if wait_timeout is not None else None

        sample_rate = 16_000
        high_pass = SpeechHighPass(sample_rate)
        integrity = getattr(self._network_camera, "audio_integrity", None)
        fault_count = integrity[1] if integrity is not None else None
        pre_roll = deque()
        pre_roll_frames = 0
        chunks = []
        speech_started = False
        stream_started = False
        voiced_frames = 0
        silent_frames = 0
        captured_frames = 0
        last_wait_report = 0.0
        last_audio_at = time.monotonic()
        print(f"ESP32 speech listening: waiting for speech, up to {seconds:g}s per utterance...", flush=True)
        while stop_event is None or not stop_event.is_set():
            if not speech_started and wait_deadline is not None and time.monotonic() >= wait_deadline:
                return None
            try:
                chunk = self._network_camera.read_audio(timeout=1.0)
            except CameraError:
                now = time.monotonic()
                if speech_started and now - last_audio_at >= 1.5:
                    raise IncompleteSpeechError("ESP32 microphone audio stopped during speech; incomplete command discarded")
                if now - last_wait_report >= 3.0:
                    print("Waiting for ESP32 microphone audio...", flush=True)
                    last_wait_report = now
                continue
            integrity = getattr(self._network_camera, "audio_integrity", None)
            if require_complete and integrity is not None and integrity[0] == "legacy":
                raise CameraError("ESP32 microphone packets have no sequence numbers; flash thirdeye_ai_device.ino")
            if integrity is not None and fault_count is not None and integrity[1] != fault_count:
                if speech_started:
                    raise IncompleteSpeechError("ESP32 microphone audio gap; incomplete command discarded")
                pre_roll.clear()
                pre_roll_frames = 0
                high_pass = SpeechHighPass(sample_rate)
                self._network_camera.clear_audio_buffer()
                fault_count = integrity[1]
                continue
            last_audio_at = time.monotonic()
            if len(chunk) < 2:
                continue
            samples = high_pass.process(np.frombuffer(chunk[: len(chunk) & ~1], dtype="<i2"))
            peak = int(np.abs(samples.astype(np.int32)).max(initial=0))
            rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
            active = peak >= voice_peak_threshold and rms >= voice_rms_threshold
            # Measure VAD before gain; both streaming ASR and local wake receive clean, louder PCM.
            samples = np.clip(samples.astype(np.int32) * 4, -32768, 32767).astype(np.int16)
            if not speech_started:
                pre_roll.append(samples.copy())
                pre_roll_frames += len(samples)
                while pre_roll_frames > sample_rate * 0.6 and len(pre_roll) > 1:
                    pre_roll_frames -= len(pre_roll.popleft())
                if not active:
                    continue
                speech_started = True
                print(f"ESP32 voice detected: peak={peak}, rms={rms:.1f}", flush=True)
                chunks.extend(pre_roll)
                captured_frames = pre_roll_frames
                voiced_frames = len(samples)
                continue
            chunks.append(samples.copy())
            captured_frames += len(samples)
            if active:
                voiced_frames += len(samples)
                silent_frames = 0
            else:
                silent_frames += len(samples)
            if on_chunk is not None and not stream_started and voiced_frames >= sample_rate * min_voiced_seconds:
                on_chunk(np.concatenate(chunks), sample_rate)
                stream_started = True
            elif on_chunk is not None and stream_started:
                on_chunk(samples, sample_rate)
            if captured_frames < seconds * sample_rate and silent_frames < sample_rate * 0.45:
                continue
            if require_complete and silent_frames < sample_rate * 0.45:
                raise IncompleteSpeechError("Speech time limit reached; incomplete command discarded")
            if voiced_frames < sample_rate * min_voiced_seconds:
                pre_roll.clear()
                pre_roll_frames = 0
                chunks.clear()
                speech_started = False
                stream_started = False
                voiced_frames = silent_frames = captured_frames = 0
                continue
            break
        else:
            return None
        integrity = getattr(self._network_camera, "audio_integrity", None)
        if integrity is not None and fault_count is not None and integrity[1] != fault_count:
            raise IncompleteSpeechError("ESP32 microphone audio gap; incomplete command discarded")
        mono = np.concatenate(chunks)
        peak = int(np.abs(mono.astype(np.int32)).max(initial=0))
        rms = float(np.sqrt(np.mean(mono.astype(np.float32) ** 2)))
        print(f"ESP32 speech audio (filtered, 4x gain): {len(mono) / sample_rate:.2f}s, peak={peak}, rms={rms:.1f}", flush=True)
        if return_samples:
            return mono
        prepared = _prepare_speech_samples(mono, sample_rate)
        with tempfile.NamedTemporaryFile(prefix="thirdeye_esp32_", suffix=".wav", delete=False) as handle:
            path = Path(handle.name)
        import wave

        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(prepared.tobytes())
        return path

    def _wait_for_esp32_phrase(
        self,
        listener: WakeWordListener,
        phrase: str,
        seconds: float,
        stop_event: threading.Event | None = None,
        *,
        allow_wake_partial: bool = False,
    ) -> bool:
        while stop_event is None or not stop_event.is_set():
            try:
                # Short completion words need less voiced audio than full commands.
                options = {"min_voiced_seconds": 0.12} if phrase.strip().lower() == "done" else {}
                path = self._record_esp32_audio(seconds, stop_event, **options)
            except (CameraError, IncompleteSpeechError) as exc:
                print(f"Waiting for usable ESP32 audio: {exc}", flush=True)
                continue
            if path is None:
                return False
            try:
                started_at = time.monotonic()
                if listener.matches_clip(path, phrase, allow_wake_partial=allow_wake_partial):
                    print(f"Local phrase transcription: {time.monotonic() - started_at:.2f}s", flush=True)
                    return True
                print(f"Local phrase transcription: {time.monotonic() - started_at:.2f}s", flush=True)
            finally:
                path.unlink(missing_ok=True)
            print("Phrase not matched; listening again...", flush=True)
        return False

    def _speak_to_user(self, text: str, *, interrupt: bool = False) -> None:
        if self._uses_esp32_audio():
            print(f"Device cue: {text}")
            if self._device_speaker is not None:
                try:
                    self._device_speaker.say(text)
                except Exception as exc:
                    print(f"ESP32 speech output failed: {exc}", flush=True)
            return
        self.speaker.say(text, interrupt=interrupt)

    def _listen_for_esp32_done(self, stop_event: threading.Event, completed_event: threading.Event) -> None:
        listener = WakeWordListener("done", self.settings.wake_language, 1.0, model_name=self.settings.wake_model)
        listener.prepare()
        try:
            if self._wait_for_esp32_phrase(listener, "done", 2.0, stop_event):
                completed_event.set()
        except Exception as exc:
            print(f"ESP32 completion input unavailable: {exc}", flush=True)

    def _get_command(self) -> Command:
        mode = input("[V]oice or [T]ype? ").strip().lower()
        text = ""
        if mode.startswith("v"):
            try:
                audio_path = Recorder(device=self.settings.microphone_device).record()
                try:
                    text = (
                        WakeWordListener(model_name=self.settings.wake_model).transcribe_clip(audio_path)
                        if self.settings.transcribe_model == "local"
                        else self.ai.transcribe(str(audio_path))
                    )
                finally:
                    audio_path.unlink(missing_ok=True)
                print(f"You said: {text}")
            except Exception as exc:
                print(f"Voice input unavailable: {exc}")
        if not text:
            text = input("Command (for example, 'find my bottle'): ").strip()
        return self.ai.understand_command(text)

    def _load_detector(self):
        if self._detector is None:
            print("Loading YOLOE. The first run may download model files...")
            from thirdeye.detector import ObjectDetector

            self._detector = ObjectDetector(
                self.settings.detector_model,
                self.settings.confidence,
                self.settings.image_size,
                self.settings.tracker,
            )
        return self._detector

    def _load_hand_tracker(self):
        if self._hand_tracker is None:
            print("Loading local hand tracker...")
            from thirdeye.hand_tracker import HandTracker

            self._hand_tracker = HandTracker(input_scale=self.settings.hand_scale)
        return self._hand_tracker

    def test_camera(self) -> None:
        import cv2

        print("Camera test started. Press Q or Esc to stop.")
        frame_count = 0
        measured_at = time.monotonic()
        last_wait_report = 0.0
        fps = 0.0
        try:
            with self._open_camera() as camera, self._video_session():
                while True:
                    try:
                        frame = camera.read()
                    except CameraError as exc:
                        now = time.monotonic()
                        if now - last_wait_report >= 3.0:
                            print(f"Waiting for camera frame: {exc}")
                            last_wait_report = now
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            break
                        continue
                    if self.settings.mirror_hand_view:
                        frame = cv2.flip(frame, 1)
                    frame_count += 1
                    now = time.monotonic()
                    elapsed = now - measured_at
                    if elapsed >= 1.0:
                        fps = frame_count / elapsed
                        frame_count = 0
                        measured_at = now
                    height, width = frame.shape[:2]
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    brightness = float(gray.mean())
                    shadows = float((gray < 16).mean() * 100)
                    highlights = float((gray > 239).mean() * 100)
                    cv2.putText(
                        frame,
                        f"{width}x{height}  {fps:.1f} FPS",
                        (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (40, 230, 80),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"Light {brightness:.0f}  Shadow {shadows:.1f}%  Clip {highlights:.1f}%",
                        (16, 62),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (40, 230, 80),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("ThirdEye AI - Camera Test", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
        finally:
            cv2.destroyAllWindows()

    def test_esp32_speaker(self) -> None:
        if not self._uses_esp32_audio():
            raise CameraError("Set THIRDEYE_CAMERA_SOURCE=esp32 before running --speaker-test")
        with self._open_camera():
            deadline = time.monotonic() + 15.0
            while not self._network_camera.speaker_http_connected and time.monotonic() < deadline:
                time.sleep(0.1)
            if not self._network_camera.speaker_http_connected:
                raise CameraError("ESP32 HTTP speaker did not poll /stream.wav within 15 seconds; flash HTTP_WAV_V1 firmware")
            if self._device_speaker is None:
                raise CameraError("ESP32 speaker output adapter is not initialized")
            print("Testing camera_down.wav via independent HTTP WAV download...", flush=True)
            self._device_speaker.play_wav(self.settings.cue_audio_dir / "camera_down.wav")
            print("ESP32 reported playback complete; verify audible Camera down.", flush=True)

    def test_esp32_imu(self) -> None:
        if self.settings.camera_source != "esp32":
            raise CameraError("Set THIRDEYE_CAMERA_SOURCE=esp32 before running --imu-test")
        print("Fall event monitor started. Ctrl+C to stop. No AI models loaded.")
        with self._open_camera():
            while True:
                monitor = self._network_camera.imu_monitor
                status = monitor["status"] or {}
                age = monitor["age"]
                healthy = monitor["connected"] and age is not None and age < 25 and status.get("sensor_ok", False)
                age_text = "none" if age is None else f"{age:.1f}s"
                print(f"IMU connected={monitor['connected']} healthy={healthy} heartbeat_age={age_text} "
                      f"samples={status.get('samples', 0)} read_failures={status.get('read_failures', 0)} "
                      f"queue_drops={status.get('queue_drops', 0)} falls={monitor['fall_count']}", flush=True)
                time.sleep(2)

    def test_esp32_device(self) -> None:
        import cv2
        import numpy as np

        if self.settings.camera_source != "esp32":
            raise CameraError("Set THIRDEYE_CAMERA_SOURCE=esp32 before running --device-test")
        print("ESP32 device test started. Press Q or Esc to stop.")
        microphone_peak = 0
        with self._open_camera() as camera, self._video_session():
            while True:
                try:
                    pcm = self._network_camera.read_audio(timeout=0.01)
                    microphone_peak = int(np.abs(np.frombuffer(pcm, dtype=np.int16).astype(np.int32)).max(initial=0))
                except CameraError:
                    pass
                try:
                    frame = camera.read()
                except CameraError:
                    continue
                imu = self._network_camera.latest_imu or {}
                monitor = self._network_camera.imu_monitor
                status = monitor["status"]
                imu_text = "IMU %.2f %.2f %.2f" % (imu.get("ax", 0), imu.get("ay", 0), imu.get("az", 0))
                if status is not None:
                    healthy = monitor["connected"] and monitor["age"] < 25 and status["sensor_ok"]
                    imu_text = f"Fall monitor {'ready' if healthy else 'unavailable'} | events {monitor['fall_count']}"
                elif not imu:
                    imu_text = "IMU waiting for heartbeat"
                cv2.putText(frame, f"Mic peak {microphone_peak}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 230, 80), 2)
                cv2.putText(
                    frame,
                    imu_text,
                    (16, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (40, 230, 80),
                    2,
                )
                cv2.imshow("ThirdEye AI - ESP32 Device Test", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        cv2.destroyAllWindows()

    def test_esp32_microphone(self, seconds: float = 5.0) -> None:
        import numpy as np
        import wave

        if self.settings.camera_source != "esp32":
            raise CameraError("Set THIRDEYE_CAMERA_SOURCE=esp32 before running --mic-test")
        with self._open_camera():
            self._network_camera.clear_audio_buffer()
            print("Waiting up to 15 seconds for the first ESP32 microphone packet...", flush=True)
            try:
                first_chunk = self._network_camera.read_audio(timeout=15.0)
            except CameraError as exc:
                status = "connected, but no audio packets arrived" if self._network_camera.audio_connected else "not connected"
                raise CameraError(f"ESP32 microphone {status}; check firmware serial status and server address") from exc
            integrity_before = getattr(self._network_camera, "audio_integrity", None)
            print(f"Recording ESP32 microphone for {seconds:g} seconds. Speak now...", flush=True)
            chunks = [first_chunk]
            started = time.monotonic()
            while time.monotonic() - started < seconds:
                try:
                    chunks.append(self._network_camera.read_audio(timeout=0.25))
                except CameraError:
                    pass
        pcm = b"".join(chunks)
        if not pcm:
            raise CameraError("No ESP32 microphone audio received")
        samples = np.frombuffer(pcm[: len(pcm) & ~1], dtype="<i2").astype(np.int32)
        peak = int(np.abs(samples).max(initial=0))
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        clipped = float(np.mean(np.abs(samples) >= 32760) * 100)
        path = Path(tempfile.gettempdir()) / f"thirdeye_mic_test_{int(time.time())}.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16_000)
            wav.writeframes(pcm)
        print(
            f"ESP32 microphone: received={len(samples) / 16_000:.2f}s / {seconds:.2f}s, "
            f"packets={len(chunks)}, peak={peak}, rms={rms:.1f}, clipped={clipped:.1f}%",
            flush=True,
        )
        integrity_after = getattr(self._network_camera, "audio_integrity", None)
        if integrity_before is not None and integrity_after is not None:
            print(
                f"ESP32 audio transport: protocol={integrity_after[0]}, "
                f"gaps={integrity_after[2] - integrity_before[2]}, "
                f"buffer_overflows={integrity_after[3] - integrity_before[3]}",
                flush=True,
            )
        print(f"Raw microphone recording: {path}", flush=True)

    def find_object(
        self,
        target: str | None = None,
        completion_listener: WakeWordListener | None = None,
    ) -> None:
        import cv2

        if not target:
            command = self._get_command()
            if command.intent is not Intent.FIND_OBJECT or not command.target:
                print("I could not identify an object to find.")
                self._speak_to_user("I could not identify an object to find.")
                return
            target = command.target
        detector = self._load_detector()
        hand_tracker = self._load_hand_tracker()
        finder = RepoObjectFinder(detector, hand_tracker)
        finder.set_target(target)
        camera_worker = self._start_camera_worker()
        announcer = RepoGuidanceAnnouncer(
            repeat_interval_s=self.settings.guidance_repeat_s,
        )
        if completion_listener is not None:
            completion_listener.unload_model()
        if self._uses_esp32_audio():
            listener_start = None
            listener_stop = threading.Event()
            done_detected = threading.Event()
            listener_process = threading.Thread(
                target=self._listen_for_esp32_done,
                args=(listener_stop, done_detected),
                name="thirdeye-esp32-done-listener",
                daemon=True,
            )
        else:
            process_context = multiprocessing.get_context("spawn")
            listener_start = process_context.Event()
            listener_stop = process_context.Event()
            done_detected = process_context.Event()
            listener_process = process_context.Process(
                target=_listen_for_done,
                args=(
                    self.settings.wake_language,
                    self.settings.microphone_device,
                    self.settings.wake_model,
                    listener_stop,
                    done_detected,
                    listener_start,
                ),
                name="thirdeye-done-listener",
                daemon=True,
            )
        with self._video_session(profile="fast"):
            listener_process.start()
            last_frame_id = -1
            camera_wait_reported = False
            analysis_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="thirdeye-finding")
            pending = None
            packet = None
            analyzed_image = None
            analyzed_at = 0.0
            last_submitted_id = -1
            displayed_frames = analyzed_frames = 0
            report_started = time.monotonic()
            report_frame_id = None
            inference_ms = 0.0
            try:
                print(
                    f"Finding {target}. Put your hand in the camera view, then say 'done' after you have it."
                )
                try:
                    camera_worker.read(timeout=0.25)
                except CameraError:
                    pass
                self._speak_to_user(f"Finding {target}. Say done to finish.")
                if listener_start is not None:
                    self.speaker.wait()
                    time.sleep(0.3)
                    listener_start.set()
                completed = False
                while not completed:
                    if done_detected.is_set():
                        completed = True
                        break
                    # Poll inference without blocking the live display or keyboard.
                    if pending is not None and pending.done():
                        result = pending.result()
                        inference_ms = (time.monotonic() - analysis_started) * 1000
                        from thirdeye.repo_display import annotate_repo_finding
                        analyzed_image = annotate_repo_finding(submitted_packet.image.copy(), result, target)
                        analyzed_at = submitted_packet.timestamp
                        analyzed_frames += 1
                        phrase = announcer.update(result.guidance)
                        if phrase is not None:
                            print(phrase)
                            self.cue_player.play(phrase)
                        pending = None
                    new_frame = False
                    try:
                        packet = camera_worker.read(last_frame_id, timeout=0.03)
                        last_frame_id = packet.frame_id
                        if report_frame_id is None:
                            report_frame_id = last_frame_id
                        new_frame = True
                        camera_wait_reported = False
                    except CameraError as exc:
                        stale = packet is None or time.monotonic() - packet.timestamp >= self.settings.camera_timeout
                        if stale and not camera_wait_reported:
                            print(f"Waiting for camera frames: {exc}")
                            camera_wait_reported = True
                    if packet is not None:
                        # At most one inference: never queue old camera frames.
                        if pending is None and packet.frame_id != last_submitted_id:
                            submitted_packet = packet
                            last_submitted_id = packet.frame_id
                            analysis_started = time.monotonic()
                            pending = analysis_pool.submit(finder.process, packet.image)
                        from thirdeye.repo_display import finding_preview
                        preview = finding_preview(packet.image, analyzed_image,
                                                  time.monotonic() - analyzed_at if analyzed_image is not None else None)
                        cv2.imshow("ThirdEye AI - Grasp Guidance", preview)
                        displayed_frames += int(new_frame)
                    now = time.monotonic()
                    elapsed = now - report_started
                    if elapsed >= 3.0:
                        received = max(0, last_frame_id - report_frame_id) if report_frame_id is not None else 0
                        print(f"Finding perf: receive_fps={received / elapsed:.1f} "
                              f"display_fps={displayed_frames / elapsed:.1f} "
                              f"analysis_fps={analyzed_frames / elapsed:.1f} "
                              f"analysis_ms={inference_ms:.0f}", flush=True)
                        report_started, report_frame_id = now, last_frame_id if packet is not None else None
                        displayed_frames = analyzed_frames = 0
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                if completed:
                    print("User said: done")
                    self._speak_to_user("Done. Object finding complete.", interrupt=True)
            finally:
                listener_stop.set()
                # Finish the single native inference before another task reuses the models.
                analysis_pool.shutdown(wait=True, cancel_futures=True)
                if listener_start is not None:
                    listener_start.set()
                listener_process.join(timeout=3)
                if listener_process.is_alive() and not self._uses_esp32_audio():
                    listener_process.terminate()
                    listener_process.join(timeout=2)
                cv2.destroyAllWindows()

    def describe_scene(
        self,
        mode: SceneMode | None = None,
        question: str | None = None,
    ) -> None:
        import cv2

        if not self.ai.online:
            raise AIUnavailable("OPENAI_API_KEY is required for scene description")
        if mode is None:
            print("\nScene mode")
            print("1. Quick navigation overview")
            print("2. Detailed scene")
            print("3. Read visible text")
            selected = input("Select [1]: ").strip() or "1"
            mode = {
                "1": SceneMode.QUICK,
                "2": SceneMode.DETAILED,
                "3": SceneMode.READ_TEXT,
            }.get(selected, SceneMode.QUICK)
            question = input("Optional question (press Enter to skip): ").strip() or None

        if question:
            self.answer_visual_question(question)
            return

        frame_count = 5 if mode is SceneMode.READ_TEXT else 3 if mode is SceneMode.DETAILED else 2
        started = time.monotonic()
        frame, quality = self._capture_text_frame() if mode is SceneMode.READ_TEXT else self._capture_scene_frame(count=frame_count)
        print(f"Scene frame: {time.monotonic() - started:.2f}s", flush=True)
        print(f"Selected frame sharpness: {quality.sharpness:.1f}; brightness: {quality.brightness:.1f}")
        if quality.warnings:
            print("Camera warning: " + ", ".join(quality.warnings))
        cv2.imshow("ThirdEye AI - Selected Scene", frame)
        cv2.waitKey(1)
        try:
            detections = []
            if mode is not SceneMode.READ_TEXT:
                detector = self._load_detector()
                detector.set_classes(SCENE_CLASSES)
                detections = detector.detect(frame)
            print(f"Scene detection: {time.monotonic() - started:.2f}s total", flush=True)
            ai_started = time.monotonic()
            analysis = self.ai.describe_scene(
                frame,
                mode=mode,
                question=question,
                local_detections=detections,
                quality=quality,
            )
            print(f"Scene AI: {time.monotonic() - ai_started:.2f}s", flush=True)
        finally:
            cv2.destroyAllWindows()
        description = render_scene_analysis(analysis, mode)
        print(f"Scene: {description}")
        self._speak_to_user(description)

    def answer_visual_question(self, question: str) -> None:
        if not self.ai.online:
            raise AIUnavailable("OPENAI_API_KEY is required for visual questions")
        needs_text_detail = is_text_question(question)
        started = time.monotonic()
        frame, quality = self._capture_text_frame() if needs_text_detail else self._capture_scene_frame(count=2)
        print(f"Visual frame: {time.monotonic() - started:.2f}s", flush=True)
        resolved = self._resolve_visual_followup(question, frame)
        if resolved is None:
            self._visual_context = None
            self._speak_to_user("I lost track of the object. Please ask the full question again.")
            return
        print(f"Visual question: {question}")
        height, width = frame.shape[:2]
        print(f"Vision frame: {width}x{height}, sharpness={quality.sharpness:.1f}, brightness={quality.brightness:.1f}", flush=True)
        if needs_text_detail and (width < 640 or height < 480):
            print("Text capture is below VGA resolution; check ESP32 PSRAM and flashed camera profile.", flush=True)
        ai_started = time.monotonic()
        answer = self.ai.answer_visual_question(frame, resolved, quality)
        print(f"Visual AI: {time.monotonic() - ai_started:.2f}s", flush=True)
        print(f"Visual answer: {answer.answer}; evidence: {answer.evidence}")
        spoken = (
            answer.answer
            if answer.can_answer and answer.evidence and answer.answer
            else "I cannot tell from this camera view. Please adjust the camera and try again."
        )
        self._speak_to_user(spoken)
        if answer.can_answer and answer.evidence:
            subject = answer.subject or (self._visual_context[0] if resolved != question else "")
            if subject and len(subject) <= 80 and "\n" not in subject:
                self._visual_context = (subject, time.monotonic(), self._visual_thumbnail(frame))
                return
        self._visual_context = None

    @staticmethod
    def _visual_thumbnail(frame):
        import cv2

        return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 48))

    def _resolve_visual_followup(self, question: str, frame) -> str | None:
        if not re.search(r"\b(it|its)\b", question, re.IGNORECASE):
            return question
        context = getattr(self, "_visual_context", None)
        if context is None or time.monotonic() - context[1] > 15:
            return None
        import numpy as np

        current = self._visual_thumbnail(frame)
        change = float(np.abs(current.astype(np.int16) - context[2].astype(np.int16)).mean())
        if change > 45:
            return None
        subject = context[0]
        return re.sub(
            r"\b(its|it)\b",
            lambda match: subject + "'s" if match.group().lower() == "its" else subject,
            question,
            flags=re.IGNORECASE,
        )

    def run_voice_control(self) -> None:
        if not self.ai.online:
            raise AIUnavailable("OPENAI_API_KEY is required for voice control")
        realtime = None
        if self.settings.transcribe_model == "gpt-live-transcribe":
            if self._uses_esp32_audio():
                def capture_esp32(_device, **kwargs):
                    return self._record_esp32_audio(
                        kwargs["duration"], kwargs["stop_event"], kwargs["wait_timeout"],
                        on_chunk=kwargs["on_chunk"],
                        require_complete=kwargs["require_complete"],
                        return_samples=True,
                    )
                realtime = RealtimeTranscriber(
                    self.settings.openai_api_key, capture_utterance=capture_esp32,
                )
            else:
                realtime = RealtimeTranscriber(self.settings.openai_api_key, self.settings.microphone_device)
        listener = WakeWordListener(
            self.settings.wake_word,
            self.settings.wake_language,
            self.settings.wake_confidence,
            self.settings.microphone_device,
            self.settings.wake_model,
        )
        self._preload_realtime_navigation()
        listener.prepare()
        if self._uses_esp32_audio() and self._network_camera is not None:
            self._network_camera.clear_audio_buffer()
        print(f"Waiting for wake word: {self.settings.wake_word}")
        conversation_active = False
        while True:
            if not conversation_active and realtime is not None:
                try:
                    realtime.prepare()
                except Exception as exc:
                    print(f"Realtime transcription prewarm unavailable: {exc}", flush=True)
            if not self._uses_esp32_audio():
                self.speaker.wait()
            following_up = conversation_active
            if following_up:
                print("Listening for a follow-up question (30 seconds)...", flush=True)
            elif self._uses_esp32_audio():
                if not self._wait_for_esp32_phrase(
                    listener,
                    self.settings.wake_word,
                    4.0,
                    allow_wake_partial=True,
                ):
                    return
                if realtime is not None:
                    try:
                        realtime.prepare()
                    except Exception as exc:
                        print(f"OpenAI realtime connection failed: {exc}", flush=True)
                        continue
                self._speak_to_user("Hi.")
                self._network_camera.clear_audio_buffer()
            else:
                time.sleep(0.4)
                listener.wait()
                self.speaker.say("Hi.")
                if realtime is not None:
                    try:
                        realtime.prepare()
                    except Exception as exc:
                        print(f"OpenAI realtime connection failed: {exc}", flush=True)
                        continue
                self.speaker.wait()
            try:
                conversation_active = True
                wait_timeout = 30.0 if following_up else None
                if realtime is not None:
                    text = realtime.listen_text(wait_timeout=wait_timeout)
                elif not self._uses_esp32_audio() and self.settings.transcribe_model == "local":
                    text = listener.listen_text(wait_timeout=wait_timeout)
                else:
                    if self._uses_esp32_audio():
                        audio_path = (
                            self._record_esp32_audio(5.0, wait_timeout=wait_timeout)
                            if following_up else self._record_esp32_audio(5.0)
                        )
                    else:
                        recorder = Recorder(device=self.settings.microphone_device)
                        audio_path = recorder.record(wait_timeout=wait_timeout) if following_up else recorder.record()
                    text = None
                    if audio_path is not None:
                        transcription_started = time.monotonic()
                        try:
                            text = (
                                listener.transcribe_clip(audio_path)
                                if self.settings.transcribe_model == "local"
                                else self.ai.transcribe(str(audio_path))
                            )
                        finally:
                            audio_path.unlink(missing_ok=True)
                        print(
                            f"Command transcription ({self.settings.transcribe_model}): "
                            f"{time.monotonic() - transcription_started:.2f}s",
                            flush=True,
                        )
                if text is None:
                    conversation_active = False
                    self._visual_context = None
                    print(f"Follow-up timed out. Waiting for wake word: {self.settings.wake_word}")
                    continue
                if not text.strip():
                    continue
                if re.sub(r"[^a-z0-9 ]", "", text.lower()).strip() == "done":
                    print(f"You said: {text}")
                    self._visual_context = None
                    conversation_active = False
                    self._speak_to_user("Standing by")
                    print(f"Waiting for wake word: {self.settings.wake_word}")
                    continue
                if not is_english_command_text(text):
                    print("Rejected non-English speech.")
                    self._speak_to_user("Please speak English.")
                    continue
                command_started = time.monotonic()
                command = self.ai.understand_command(text)
                print(f"Command routing: {time.monotonic() - command_started:.2f}s", flush=True)
                print(f"You said: {text}")
                if command.intent is Intent.EXIT:
                    if realtime is not None:
                        realtime.close()
                    self._speak_to_user("Goodbye")
                    return
                if command.intent is Intent.STANDBY:
                    self._speak_to_user("Standing by")
                    conversation_active = False
                    self._visual_context = None
                elif command.intent is Intent.FIND_OBJECT and command.target:
                    conversation_active = False
                    self._visual_context = None
                    if realtime is not None:
                        realtime.close()
                    self.find_object(command.target, listener)
                elif command.intent is Intent.DESCRIBE_SCENE:
                    self._visual_context = None
                    self.describe_scene(SceneMode.QUICK)
                    if not self._uses_esp32_audio():
                        playback_started = time.monotonic()
                        self.speaker.wait()
                        print(f"Voice playback: {time.monotonic() - playback_started:.2f}s", flush=True)
                elif command.intent is Intent.VISUAL_QUESTION:
                    self.answer_visual_question(text)
                    if not self._uses_esp32_audio():
                        playback_started = time.monotonic()
                        self.speaker.wait()
                        print(f"Voice playback: {time.monotonic() - playback_started:.2f}s", flush=True)
                else:
                    self._speak_to_user("I did not understand the command")
            except IncompleteSpeechError:
                conversation_active = True
                print("Audio interrupted; listening again.", flush=True)
            except (CameraError, AIUnavailable) as exc:
                conversation_active = False
                print(f"Cannot complete command: {exc}")
                self._speak_to_user(str(exc))
            except Exception as exc:
                conversation_active = False
                print(f"Command failed; returning to wake mode: {exc}")
                traceback.print_exc()
                self._speak_to_user("I could not complete that command. Please try again.")
            if not conversation_active:
                print(f"Waiting for wake word: {self.settings.wake_word}")

    def run(self) -> None:
        while True:
            print("\nThirdEye AI")
            print("1. Find Object")
            print("2. Describe Scene")
            print("3. Exit")
            choice = input("Select: ").strip()
            try:
                if choice == "1":
                    self.find_object()
                elif choice == "2":
                    self.describe_scene()
                elif choice == "3":
                    self.speaker.say("Goodbye")
                    return
                else:
                    print("Please choose 1, 2, or 3.")
            except (CameraError, AIUnavailable) as exc:
                print(f"Cannot complete action: {exc}")
                self.speaker.say(str(exc))
            except KeyboardInterrupt:
                print("\nAction stopped.")


def main() -> None:
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="ThirdEye AI device server")
    parser.add_argument("--check", action="store_true", help="check dependencies, API key, GPU, and camera")
    parser.add_argument("--camera-test", action="store_true", help="show the camera stream without loading AI models")
    parser.add_argument("--device-test", action="store_true", help="show ESP32 camera, microphone and IMU data")
    parser.add_argument("--imu-test", action="store_true", help="monitor ESP32 fall events and sensor heartbeats without AI")
    parser.add_argument("--mic-test", action="store_true", help="record raw ESP32 microphone audio for diagnosis")
    parser.add_argument("--speaker-test", action="store_true", help="verify audio uplink then send camera_down.wav to ESP32")
    parser.add_argument("--voice-control", action="store_true", help="listen for the ThirdEye wake word")
    parser.add_argument("--target", help="start object finding immediately with this target")
    args = parser.parse_args()
    settings = Settings.load()
    if args.check:
        from thirdeye.diagnostics import run_checks

        raise SystemExit(0 if run_checks(settings) else 1)
    app = ThirdEyeApp(settings)
    try:
        if args.voice_control:
            app.run_voice_control()
        elif args.device_test:
            app.test_esp32_device()
        elif args.imu_test:
            app.test_esp32_imu()
        elif args.mic_test:
            app.test_esp32_microphone()
        elif args.speaker_test:
            app.test_esp32_speaker()
        elif args.camera_test:
            app.test_camera()
        elif args.target:
            app.find_object(args.target)
        else:
            app.run()
    except KeyboardInterrupt:
        print("\nThirdEye AI stopped.")
        sys.exit(130)
    finally:
        app.close()
