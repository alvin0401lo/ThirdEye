from __future__ import annotations

import json
import math
import re
import asyncio
import io
import wave
from contextlib import asynccontextmanager, contextmanager, suppress
from collections import deque
import threading
import time
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from thirdeye.camera import CameraError


def is_valid_camera_jpeg(data: bytes) -> bool:
    return 4 <= len(data) <= 2_000_000 and data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")


def is_valid_audio_pcm(data: bytes) -> bool:
    return 2 <= len(data) <= 4096 and len(data) % 2 == 0


def decode_microphone_packet(data: bytes) -> tuple[int | None, bytes] | None:
    if data.startswith(b"TEA1"):
        pcm = data[8:]
        if len(data) < 10 or not is_valid_audio_pcm(pcm):
            return None
        return int.from_bytes(data[4:8], "little"), pcm
    if is_valid_audio_pcm(data):
        return None, data
    return None


class Esp32Camera:
    """Receives the camera, microphone PCM and IMU data from one ESP32 device."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8081,
        timeout: float = 5.0,
        token: str | None = None,
        companion_bridge_url: str | None = None,
        companion_room: str = "THIRDEYE",
    ) -> None:
        import uvicorn

        self._timeout = timeout
        self._companion_bridge_url = companion_bridge_url
        self._companion_room = companion_room
        self._condition = threading.Condition()
        self._audio_condition = threading.Condition()
        self._latest_frame: Any | None = None
        self._latest_audio = b""
        self._audio_chunks: deque[bytes] = deque(maxlen=1200)
        self._latest_imu: dict[str, float] | None = None
        self._imu_lock = threading.Lock()
        self._imu_status: dict[str, Any] | None = None
        self._imu_received_at: float | None = None
        self._fall_ids: deque[str] = deque(maxlen=256)
        self._fall_count = 0
        self._latest_fall: dict[str, Any] | None = None
        self._sequence = 0
        self._last_read_sequence = 0
        self._audio_sequence = 0
        self._last_read_audio_sequence = 0
        self._audio_expected_sequence: int | None = None
        self._audio_protocol: str | None = None
        self._audio_fault_count = 0
        self._audio_gap_count = 0
        self._audio_overflow_count = 0
        self._connected = False
        self._audio_connected = False
        self._audio_firmware: str | None = None
        self._audio_socket: WebSocket | None = None
        self._audio_accept_lock = asyncio.Lock()
        self._audio_last_received_at = 0.0
        self._http_speaker_condition = threading.Condition()
        self._http_speaker_lock = threading.Lock()
        self._http_speaker_seen = 0.0
        self._http_speaker_id = 0
        self._http_speaker_job = None
        self._imu_connected = False
        self._camera_loop: asyncio.AbstractEventLoop | None = None
        self._camera_commands: asyncio.Queue[str] | None = None
        self._video_users = 0
        self._companion_viewing = False
        self._video_enabled = False
        self._still_lock = threading.Lock()
        self._still_id = 0
        self._still_result_id = 0
        self._still_frame: Any | None = None
        self._still_error = False
        relay_frames: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)

        @asynccontextmanager
        async def lifespan(app):
            relay = asyncio.create_task(self._relay_camera_frames(relay_frames)) if self._companion_bridge_url else None
            try:
                yield
            finally:
                if relay is not None:
                    relay.cancel()
                    with suppress(asyncio.CancelledError):
                        await relay

        app = FastAPI(lifespan=lifespan)

        @app.get("/stream.wav")
        async def speaker_wav(request: Request):
            if token and request.query_params.get("token") != token:
                return Response(status_code=403)
            with self._http_speaker_condition:
                self._http_speaker_seen = time.monotonic()
                self._http_speaker_condition.notify_all()
                job = self._http_speaker_job
                if job is None or job["served"]:
                    return Response(status_code=204)
                job["served"] = True
                body, request_id = job["wav"], job["id"]
            print(f"ESP32 HTTP WAV requested: id={request_id}, bytes={len(body)}; waiting for board completion", flush=True)
            return Response(body, media_type="audio/wav", headers={
                "X-Playback-Id": str(request_id), "Cache-Control": "no-store",
            })

        @app.post("/speaker/status")
        async def speaker_status(request: Request):
            if token and request.query_params.get("token") != token:
                return Response(status_code=403)
            request_id = request.query_params.get("id", "")
            state = request.query_params.get("state", "")
            with self._http_speaker_condition:
                job = self._http_speaker_job
                if job is None or request_id != str(job["id"]) or not job["served"]:
                    return Response(status_code=409)
                if state not in ("done", "error"):
                    return Response(status_code=400)
                if job["result"] is None:
                    job["result"] = state
                self._http_speaker_condition.notify_all()
            return Response(status_code=204)

        @app.websocket("/ws/camera")
        async def receive_camera(websocket: WebSocket):
            if token and websocket.query_params.get("token") != token:
                await websocket.close(code=1008)
                return
            if self._connected:
                await websocket.close(code=1013)
                return
            await websocket.accept()
            self._connected = True
            self._camera_loop = asyncio.get_running_loop()
            self._camera_commands = asyncio.Queue()
            self._camera_commands.put_nowait("VIDEO:ON" if self._video_enabled else "VIDEO:OFF")
            print("ESP32 camera connected")

            async def send_camera_commands() -> None:
                while True:
                    command = await self._camera_commands.get()
                    await websocket.send_text(command)

            sender = asyncio.create_task(send_camera_commands())
            expected_still_id = 0
            connected_at = time.monotonic()
            close_code, close_reason = None, "handler stopped"
            try:
                while True:
                    packet = await websocket.receive()
                    if packet["type"] == "websocket.disconnect":
                        close_code = packet.get("code")
                        close_reason = packet.get("reason", "")
                        break
                    message = packet.get("text")
                    if message is not None:
                        if message.startswith("SNAP:") and message[5:].isdigit():
                            expected_still_id = int(message[5:])
                        elif message.startswith("SNAP_ERROR:") and message[11:].isdigit():
                            with self._condition:
                                self._still_result_id = int(message[11:])
                                self._still_error = True
                                self._condition.notify_all()
                        continue
                    jpeg = packet.get("bytes")
                    if jpeg is None:
                        continue
                    if not is_valid_camera_jpeg(jpeg):
                        continue
                    frame = await asyncio.to_thread(self._decode, jpeg)
                    if frame is None:
                        continue
                    with self._condition:
                        if expected_still_id:
                            self._still_frame = frame
                            self._still_result_id = expected_still_id
                            self._still_error = frame.shape[:2] != (1200, 1600)
                            expected_still_id = 0
                        else:
                            self._latest_frame = frame
                            self._sequence += 1
                            if self._companion_bridge_url:
                                if relay_frames.full():
                                    relay_frames.get_nowait()
                                relay_frames.put_nowait(jpeg)
                        self._condition.notify_all()
            except WebSocketDisconnect as exc:
                close_code, close_reason = exc.code, exc.reason
            finally:
                sender.cancel()
                self._connected = False
                self._camera_loop = None
                self._camera_commands = None
                with self._condition:
                    self._condition.notify_all()
                print(f"ESP32 camera disconnected: code={close_code}, reason={close_reason[:160]!r}, "
                      f"connected_for={time.monotonic() - connected_at:.1f}s", flush=True)

        @app.websocket("/ws/audio")
        async def receive_audio(websocket: WebSocket):
            if token and websocket.query_params.get("token") != token:
                print("ESP32 audio rejected: missing or incorrect device token", flush=True)
                await websocket.close(code=1008)
                return
            async with self._audio_accept_lock:
                old_socket = self._audio_socket
                if old_socket is not None:
                    same_device = (old_socket.client is not None and websocket.client is not None
                                   and old_socket.client.host == websocket.client.host)
                    stale = self._audio_firmware is None and time.monotonic() - self._audio_last_received_at >= 5.0
                    if not same_device or not stale:
                        print("ESP32 audio rejected: another active audio channel is connected", flush=True)
                        await websocket.close(code=1013)
                        return
                    # Retire ownership first so old cleanup cannot disconnect the replacement.
                    self._audio_socket = None
                    with self._audio_condition:
                        self._audio_connected = False
                        self._audio_fault_count += 1
                        self._audio_condition.notify_all()
                    with suppress(Exception):
                        await old_socket.close(code=1012)
                    print("ESP32 audio retiring stalled, unready connection from the same device", flush=True)
                await websocket.accept()
                self._audio_socket = websocket
                self._audio_last_received_at = time.monotonic()
                with self._audio_condition:
                    self._audio_connected = True
                    self._audio_expected_sequence = None
                    self._audio_protocol = None
                    self._audio_firmware = None
                    self._audio_chunks.clear()
            print("ESP32 audio socket accepted; waiting for board protocol readiness", flush=True)
            connected_at = time.monotonic()
            close_code, close_reason = None, "handler stopped"
            try:
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        close_code = message.get("code")
                        close_reason = message.get("reason", "")
                        break
                    if self._audio_socket is not websocket:
                        close_reason = "replaced by another connection"
                        break
                    if message.get("text") is not None:
                        if message["text"].startswith("DEVICE_AUDIO:"):
                            with self._audio_condition:
                                self._audio_firmware = message["text"].partition(":")[2]
                                self._audio_last_received_at = time.monotonic()
                                self._audio_condition.notify_all()
                            print(f"ESP32 audio firmware reported: {message['text']}", flush=True)
                            continue
                        continue
                    packet = decode_microphone_packet(message.get("bytes") or b"")
                    with self._audio_condition:
                        if packet is None:
                            self._audio_fault_count += 1
                            continue
                        sequence, pcm = packet
                        self._audio_last_received_at = time.monotonic()
                        if sequence is None:
                            self._audio_protocol = "legacy"
                            self._audio_expected_sequence = None
                        else:
                            if self._audio_protocol == "legacy":
                                self._audio_fault_count += 1
                            self._audio_protocol = "sequenced"
                            expected = self._audio_expected_sequence
                            if expected is not None and sequence != expected:
                                gap = (sequence - expected) & 0xFFFFFFFF
                                self._audio_gap_count += gap if gap < 0x80000000 else 1
                                self._audio_fault_count += 1
                            self._audio_expected_sequence = (sequence + 1) & 0xFFFFFFFF
                        if len(self._audio_chunks) == self._audio_chunks.maxlen:
                            self._audio_overflow_count += 1
                            self._audio_fault_count += 1
                        self._latest_audio = pcm
                        self._audio_chunks.append(pcm)
                        self._audio_sequence += 1
                        self._audio_condition.notify_all()
            except WebSocketDisconnect as exc:
                close_code, close_reason = exc.code, exc.reason
            finally:
                if self._audio_socket is websocket:
                    self._audio_socket = None
                    with self._audio_condition:
                        self._audio_connected = False
                        self._audio_expected_sequence = None
                        self._audio_fault_count += 1
                        self._audio_condition.notify_all()
                print(f"ESP32 microphone disconnected: code={close_code}, reason={close_reason[:160]!r}, "
                      f"connected_for={time.monotonic() - connected_at:.1f}s", flush=True)

        @app.websocket("/ws/imu")
        async def receive_imu(websocket: WebSocket):
            if token and websocket.query_params.get("token") != token:
                await websocket.close(code=1008)
                return
            if self._imu_connected:
                await websocket.close(code=1013)
                return
            await websocket.accept()
            self._imu_connected = True
            print("ESP32 IMU connected")
            try:
                while True:
                    message = await websocket.receive_text()
                    try:
                        ack = self._receive_imu_message(json.loads(message))
                        if ack is not None:
                            await websocket.send_text(ack)
                    except (KeyError, TypeError, ValueError, OverflowError, json.JSONDecodeError):
                        continue
            except WebSocketDisconnect:
                pass
            finally:
                self._imu_connected = False
                print("ESP32 IMU disconnected")

        # Compatibility diagnostic: retain transport close codes instead of SansIO's generic 1005.
        self._server = uvicorn.Server(
            uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False,
                           ws="websockets")
        )
        self._thread = threading.Thread(target=self._server.run, name="thirdeye-camera-server", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 5.0
        while not self._server.started and self._thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not self._server.started:
            raise CameraError(f"Could not start ESP32 camera server on {host}:{port}")
        print(f"Waiting for ESP32 device at ws://{host}:{port}/ws/(camera|audio|imu)")
        print("ESP32 WebSocket backend: websockets; keepalive interval=20s, timeout=20s", flush=True)

    def _receive_imu_message(self, data: Any) -> str | None:
        if not isinstance(data, dict):
            return None
        kind = data.get("type")
        if kind == "fall_suspected":
            event_id = data.get("event_id")
            if not isinstance(event_id, str) or not re.fullmatch(r"[0-9a-fA-F]{8}-[0-9]{1,10}", event_id):
                return None
            fields = ("uptime_ms", "peak_g", "peak_gyro", "tilt_deg")
            if any(type(data.get(key)) not in (int, float) or not math.isfinite(data[key]) for key in fields):
                return None
            if not (0 <= data["uptime_ms"] <= 0xFFFFFFFF and 0 <= data["peak_g"] <= 16 and
                    0 <= data["peak_gyro"] <= 1800 and 0 <= data["tilt_deg"] <= 180):
                return None
            event = {key: data[key] for key in ("type", "event_id", *fields)}
            event["received_at"] = time.time()
            with self._imu_lock:
                duplicate = event_id in self._fall_ids
                if not duplicate:
                    self._fall_ids.append(event_id)
                    self._fall_count += 1
                    self._latest_fall = event
            if not duplicate:
                print(f"[FALL SUSPECTED] id={event_id} peak={data['peak_g']:.2f}g "
                      f"tilt={data['tilt_deg']:.1f}deg; not a confirmed human fall", flush=True)
            return f"ACK:{event_id}"
        if kind == "imu_status":
            fields = ("uptime_ms", "samples", "read_failures", "queue_drops")
            if type(data.get("sensor_ok")) is not bool or any(
                type(data.get(key)) is not int or not 0 <= data[key] <= 0xFFFFFFFF for key in fields
            ):
                return None
            with self._imu_lock:
                self._imu_status = {key: data[key] for key in (*fields, "sensor_ok")}
                self._imu_received_at = time.monotonic()
            return None
        if kind is None:
            # Older diagnostic firmware can still display six-axis samples.
            values = {key: float(data[key]) for key in ("ax", "ay", "az", "gx", "gy", "gz")}
            if all(math.isfinite(value) for value in values.values()):
                self._latest_imu = values
        return None

    @property
    def imu_monitor(self) -> dict[str, Any]:
        with self._imu_lock:
            return {
                "connected": self._imu_connected,
                "status": self._imu_status.copy() if self._imu_status is not None else None,
                "age": time.monotonic() - self._imu_received_at if self._imu_received_at is not None else None,
                "fall_count": self._fall_count,
                "latest_fall": self._latest_fall.copy() if self._latest_fall is not None else None,
            }

    async def _relay_camera_frames(self, frames: asyncio.Queue[bytes]) -> None:
        from websockets.asyncio.client import connect

        unavailable_reported = False
        while True:
            try:
                async with connect(self._companion_bridge_url, open_timeout=3) as companion:
                    await companion.send(json.dumps({
                        "type": "join_camera_relay", "room": self._companion_room,
                    }))
                    print(f"Companion camera bridge connected: room={self._companion_room}", flush=True)
                    unavailable_reported = False
                    async def receive_demand():
                        async for message in companion:
                            if isinstance(message, str):
                                data = json.loads(message)
                                if data.get("type") == "camera_demand" and type(data.get("active")) is bool:
                                    self._set_companion_viewing(data["active"])

                    receiver = asyncio.create_task(receive_demand())
                    last_monitor = None
                    try:
                        while not receiver.done():
                            monitor = self.imu_monitor
                            status = monitor["status"]
                            snapshot = {
                                "type": "fall_monitor", "connected": monitor["connected"],
                                "sensor_ok": status["sensor_ok"] if status is not None else None,
                                "stale": monitor["age"] is None or monitor["age"] >= 25,
                                "fall_count": monitor["fall_count"], "latest_fall": monitor["latest_fall"],
                            }
                            if snapshot != last_monitor:
                                await companion.send(json.dumps(snapshot))
                                last_monitor = snapshot
                            try:
                                frame = await asyncio.wait_for(frames.get(), timeout=0.5)
                            except asyncio.TimeoutError:
                                continue
                            await companion.send(frame)
                        await receiver
                    finally:
                        receiver.cancel()
                        self._set_companion_viewing(False)
                        with suppress(asyncio.CancelledError):
                            await receiver
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not unavailable_reported:
                    print(f"Companion camera bridge unavailable: {exc}", flush=True)
                    unavailable_reported = True
                await asyncio.sleep(2)

    @staticmethod
    def _decode(jpeg: bytes) -> Any | None:
        import cv2
        import numpy as np

        return cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)

    def _update_video_demand(self) -> None:
        # Called with the frame condition held; tasks and website share one stream.
        enabled = self._video_users > 0 or self._companion_viewing
        if enabled == self._video_enabled:
            return
        self._video_enabled = enabled
        self._last_read_sequence = self._sequence
        self._latest_frame = None
        if self._camera_loop is not None and self._camera_commands is not None:
            commands = self._camera_commands
            self._camera_loop.call_soon_threadsafe(commands.put_nowait, "VIDEO:ON" if enabled else "VIDEO:OFF")
        print(f"ESP32 video requested: {'ON' if enabled else 'OFF'}", flush=True)

    def _set_companion_viewing(self, active: bool) -> None:
        with self._condition:
            self._companion_viewing = active
            self._update_video_demand()

    @contextmanager
    def video_session(self):
        with self._condition:
            self._video_users += 1
            self._update_video_demand()
        try:
            yield self
        finally:
            with self._condition:
                self._video_users -= 1
                self._update_video_demand()

    def read(self) -> Any:
        deadline = time.monotonic() + self._timeout
        with self._condition:
            while self._latest_frame is None or self._sequence == self._last_read_sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CameraError("No new ESP32 camera frame received within the timeout")
                self._condition.wait(remaining)
            self._last_read_sequence = self._sequence
            return self._latest_frame.copy()

    def capture_still(self, timeout: float = 8.0) -> Any:
        """Request one UXGA JPEG without replacing the live VGA frame."""
        with self._still_lock:
            with self._condition:
                loop = self._camera_loop
                commands = self._camera_commands
                if not self._connected or loop is None or commands is None:
                    raise CameraError("ESP32 camera is not connected")
                self._still_id += 1
                request_id = self._still_id
                loop.call_soon_threadsafe(commands.put_nowait, f"SNAP:{request_id}")
                deadline = time.monotonic() + timeout
                while self._still_result_id != request_id and self._connected:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise CameraError("ESP32 text photo timed out; check the flashed firmware and PSRAM")
                    self._condition.wait(remaining)
                if not self._connected:
                    raise CameraError("ESP32 camera disconnected during text photo")
                if self._still_error or self._still_frame is None:
                    raise CameraError("ESP32 could not capture a 1600x1200 text photo; check PSRAM")
                return self._still_frame.copy()

    def read_audio(self, timeout: float = 5.0) -> bytes:
        deadline = time.monotonic() + timeout
        with self._audio_condition:
            while not self._audio_chunks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CameraError("No new ESP32 microphone audio received within the timeout")
                self._audio_condition.wait(remaining)
            return self._audio_chunks.popleft()

    def clear_audio_buffer(self) -> None:
        with self._audio_condition:
            self._audio_chunks.clear()
            self._audio_expected_sequence = None

    @property
    def audio_integrity(self) -> tuple[str | None, int, int, int]:
        with self._audio_condition:
            return (
                self._audio_protocol, self._audio_fault_count,
                self._audio_gap_count, self._audio_overflow_count,
            )

    @property
    def audio_connected(self) -> bool:
        return self._audio_connected

    @property
    def speaker_http_connected(self) -> bool:
        with self._http_speaker_condition:
            return self._http_speaker_seen > 0 and time.monotonic() - self._http_speaker_seen < 5.0

    def play_speaker_pcm(self, pcm: bytes) -> None:
        if not pcm or len(pcm) % 2 or len(pcm) > 1_920_000:
            raise CameraError("Speaker PCM must be mono PCM16, nonempty and at most 60 seconds")
        stream = io.BytesIO()
        with wave.open(stream, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(pcm)
        with self._http_speaker_lock:
            with self._http_speaker_condition:
                self._http_speaker_id += 1
                job = {"id": self._http_speaker_id, "wav": stream.getvalue(),
                       "served": False, "result": None}
                self._http_speaker_job = job
                deadline = time.monotonic() + len(pcm) / 32000 + 25
                print(f"ESP32 HTTP WAV queued: id={job['id']}, seconds={len(pcm) / 32000:.2f}", flush=True)
                try:
                    while job["result"] is None:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise CameraError(f"ESP32 HTTP speaker timed out: downloaded={job['served']}; check serial HTTP speaker logs")
                        self._http_speaker_condition.wait(remaining)
                    if job["result"] != "done":
                        raise CameraError("ESP32 HTTP WAV playback failed; check serial output")
                finally:
                    self._http_speaker_job = None
            print(f"ESP32 HTTP WAV playback finished: id={job['id']}", flush=True)

    @property
    def latest_imu(self) -> dict[str, float] | None:
        return self._latest_imu.copy() if self._latest_imu is not None else None

    def release(self) -> None:
        with self._http_speaker_condition:
            if self._http_speaker_job is not None:
                self._http_speaker_job["result"] = "error"
            self._http_speaker_condition.notify_all()
        self._server.should_exit = True
        self._thread.join(timeout=3)
