import time

import numpy as np

from thirdeye.camera_worker import CameraWorker


class _FakeCamera:
    def __init__(self) -> None:
        self.value = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        self.value += 1
        time.sleep(0.005)
        return np.full((4, 4, 3), self.value, dtype=np.uint8)


def test_camera_worker_returns_a_newer_frame():
    worker = CameraWorker(_FakeCamera, "test")
    worker.start()
    try:
        first = worker.read(timeout=1)
        second = worker.read(after_frame_id=first.frame_id, timeout=1)
        assert second.frame_id > first.frame_id
    finally:
        worker.close()
