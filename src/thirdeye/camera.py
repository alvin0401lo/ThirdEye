from __future__ import annotations

from typing import Any


class CameraError(RuntimeError):
    pass


class Webcam:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        import cv2

        self._cv2 = cv2
        self.index = index
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        print(f"Opening camera index {index}...")
        self._capture = cv2.VideoCapture(index, backend)
        if hasattr(cv2, "CAP_PROP_FOURCC"):
            self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._capture.isOpened():
            self._capture.release()
            raise CameraError(
                f"Cannot open camera index {index}. "
                "Check that the external camera is connected and not in use by another app."
            )
        actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {actual_width}x{actual_height}")

    def read(self) -> Any:
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("Camera opened but did not return a frame")
        return frame

    def release(self) -> None:
        self._capture.release()

    def __enter__(self) -> "Webcam":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
