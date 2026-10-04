from types import SimpleNamespace

import numpy as np

from thirdeye.hand_tracker import HandTracker


def test_hand_tracker_uses_scaled_inference_frame():
    received_shapes = []

    class FakeImage:
        def __init__(self, image_format, data):
            del image_format
            received_shapes.append(data.shape)

    class FakeHands:
        def detect_async(self, _image, _timestamp):
            return None

    tracker = HandTracker.__new__(HandTracker)
    tracker._mp = SimpleNamespace(
        Image=FakeImage,
        ImageFormat=SimpleNamespace(SRGB="srgb"),
    )
    tracker._hands = FakeHands()
    tracker._last_timestamp_ms = 0
    tracker._last_result = SimpleNamespace(hand_landmarks=[], handedness=[])
    tracker._input_scale = 0.8

    assert tracker.track(np.zeros((100, 200, 3), dtype=np.uint8)) == []
    assert received_shapes == [(80, 160, 3)]
