from __future__ import annotations

import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from thirdeye.camera import CameraError
from thirdeye.models import FramePacket


class CameraWorker:
    """Continuously reads a camera and exposes only its newest frame."""

    def __init__(
        self,
        camera_factory: Callable[[], AbstractContextManager[Any]],
        source: str,
        mirror: bool = False,
        reconnect_delay: float = 0.5,
    ) -> None:
        self._camera_factory = camera_factory
        self._source = source
        self._mirror = mirror
        self._reconnect_delay = reconnect_delay
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._latest: FramePacket | None = None
        self._last_error: Exception | None = None
        self._frame_id = 0
        self._thread = threading.Thread(target=self._run, name="thirdeye-camera", daemon=True)

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def read(self, after_frame_id: int = -1, timeout: float = 5.0) -> FramePacket:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._latest is None or self._latest.frame_id <= after_frame_id:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = f": {self._last_error}" if self._last_error else ""
                    raise CameraError(f"No new camera frame within {timeout:g} seconds{detail}")
                self._condition.wait(remaining)
            return self._latest

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=6)

    def clear_frames(self) -> None:
        with self._condition:
            self._latest = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with self._camera_factory() as camera:
                    while not self._stop.is_set():
                        image = camera.read()
                        if self._mirror:
                            import cv2

                            image = cv2.flip(image, 1)
                        packet = FramePacket(
                            frame_id=self._frame_id,
                            timestamp=time.monotonic(),
                            image=image,
                            source=self._source,
                        )
                        self._frame_id += 1
                        with self._condition:
                            self._latest = packet
                            self._last_error = None
                            self._condition.notify_all()
            except Exception as exc:
                with self._condition:
                    self._last_error = exc
                    self._condition.notify_all()
                self._stop.wait(self._reconnect_delay)
