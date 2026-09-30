from __future__ import annotations

import base64
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from queue import Empty, Full, Queue

from websockets.sync.client import connect

from thirdeye.audio import _capture_utterance, _resample_audio


class RealtimeTranscriber:
    """One committed utterance per connection; partial text never leaves this class."""

    def __init__(self, api_key: str, device=None, capture_utterance=None):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for realtime transcription")
        self.api_key = api_key
        self.device = device
        self.capture_utterance = capture_utterance
        self._ws = None

    def prepare(self):
        if self._ws is not None:
            return
        started = time.monotonic()
        print("Connecting OpenAI realtime transcription...", flush=True)
        ws = connect(
            "wss://api.openai.com/v1/realtime?intent=transcription",
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            open_timeout=10, close_timeout=2,
        )
        try:
            ws.send(json.dumps({
                "type": "session.update",
                "session": {"type": "transcription", "audio": {"input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {
                        "model": "gpt-live-transcribe", "languages": ["en"],
                        "keywords": ["Third Eye"], "delay": "low",
                    },
                    "turn_detection": None,
                }}},
            }))
            deadline = time.monotonic() + 10
            while True:
                event = json.loads(ws.recv(timeout=max(0.01, deadline - time.monotonic())))
                self._check_error(event)
                if event["type"] == "session.updated":
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("OpenAI transcription session setup timed out")
        except Exception:
            ws.close()
            raise
        self._ws = ws
        print(f"ASR connect: {time.monotonic() - started:.2f}s", flush=True)

    def close(self):
        ws, self._ws = self._ws, None
        if ws is not None:
            ws.close()

    def listen_text(self, *, wait_timeout=None):
        wait_deadline = time.monotonic() + wait_timeout if wait_timeout is not None else None
        self.prepare()
        with nullcontext(self._ws) as ws:

            stop = threading.Event()
            packets = Queue(maxsize=100)
            errors = []

            def send_audio():
                try:
                    while not stop.is_set():
                        try:
                            packet = packets.get(timeout=0.1)
                        except Empty:
                            continue
                        if packet is None:
                            ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                            return
                        ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": packet}))
                except Exception as exc:
                    errors.append(exc)
                    stop.set()

            def receive_text():
                item_id = None
                partial = ""
                try:
                    while not stop.is_set():
                        try:
                            event = json.loads(ws.recv(timeout=0.2))
                        except TimeoutError:
                            continue
                        self._check_error(event)
                        kind = event["type"]
                        if kind == "input_audio_buffer.committed":
                            item_id = event["item_id"]
                        elif kind == "conversation.item.input_audio_transcription.delta":
                            partial += event.get("delta", "")
                            print(f"[ASR PARTIAL] {partial}", flush=True)
                        elif kind == "conversation.item.input_audio_transcription.completed":
                            if item_id is not None and event.get("item_id") == item_id:
                                return event["transcript"].strip()
                except Exception as exc:
                    errors.append(exc)
                    stop.set()
                return None

            def queue_chunk(samples, rate):
                if errors:
                    raise RuntimeError(f"Realtime transcription failed: {errors[0]}")
                pcm = _resample_audio(samples, rate, 24000).astype("<i2").tobytes()
                try:
                    packets.put_nowait(base64.b64encode(pcm).decode("ascii"))
                except Full:
                    raise RuntimeError("Audio upload is too slow; incomplete command discarded") from None

            with ThreadPoolExecutor(max_workers=2) as workers:
                sender = workers.submit(send_audio)
                receiver = workers.submit(receive_text)
                completed = False
                try:
                    print("OpenAI realtime ready. Speak now.", flush=True)
                    remaining = max(0.0, wait_deadline - time.monotonic()) if wait_deadline is not None else None
                    capture = self.capture_utterance or _capture_utterance
                    samples = capture(
                        self.device, duration=12.0, wait_timeout=remaining,
                        stop_event=stop, require_complete=True, on_chunk=queue_chunk,
                    )
                    if errors:
                        raise RuntimeError(f"Realtime transcription failed: {errors[0]}")
                    if samples is None:
                        return None
                    started = time.monotonic()
                    packets.put(None, timeout=2)
                    sender.result(timeout=10)
                    text = receiver.result(timeout=15)
                    if errors:
                        raise RuntimeError(f"Realtime transcription failed: {errors[0]}")
                    if text is None:
                        raise RuntimeError("OpenAI returned no final transcript")
                    print(f"[ASR FINAL] {text or '[no speech]'}", flush=True)
                    print(f"ASR end-to-final: {time.monotonic() - started:.2f}s", flush=True)
                    completed = True
                    return text
                finally:
                    stop.set()
                    if not completed:
                        self.close()

    @staticmethod
    def _check_error(event):
        if event.get("type") in {"error", "conversation.item.input_audio_transcription.failed"}:
            error = event.get("error", {})
            raise RuntimeError(f"OpenAI transcription: {error.get('message', 'unknown error')}")
