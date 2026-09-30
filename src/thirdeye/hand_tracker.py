from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from thirdeye.models import HandObservation, Point


class HandTracker:
    """Tracks hand landmarks locally and exposes a grasp-oriented pinch point."""

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.40,
        presence_confidence: float = 0.50,
        tracking_confidence: float = 0.70,
        model_path: str | Path | None = None,
        input_scale: float = 0.8,
    ) -> None:
        if not 0.25 <= input_scale <= 1.0:
            raise ValueError("input_scale must be between 0.25 and 1.0")
        import mediapipe as mp

        self._mp = mp
        self._input_scale = input_scale
        asset = Path(model_path) if model_path else Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"
        if not asset.exists():
            raise FileNotFoundError(f"MediaPipe hand model not found: {asset}. Run setup.ps1 again.")
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(asset)),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
            result_callback=self._on_result,
        )
        self._hands = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = 0
        self._last_result = None

    def _on_result(self, result: Any, _image: Any, _timestamp_ms: int) -> None:
        self._last_result = result

    def track(self, frame: Any) -> list[HandObservation]:
        import cv2

        height, width = frame.shape[:2]
        inference_frame = frame
        if self._input_scale < 1.0:
            inference_frame = cv2.resize(
                frame,
                None,
                fx=self._input_scale,
                fy=self._input_scale,
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(self._last_timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
        self._last_timestamp_ms = timestamp_ms
        self._hands.detect_async(image, timestamp_ms)
        result = self._last_result
        observations: list[HandObservation] = []
        if result is None or not result.hand_landmarks:
            return observations

        handedness = result.handedness or []
        for index, hand in enumerate(result.hand_landmarks):
            points = tuple(Point(mark.x * width, mark.y * height) for mark in hand)
            palm_ids = (0, 5, 9, 13, 17)
            palm = Point(
                sum(points[item].x for item in palm_ids) / len(palm_ids),
                sum(points[item].y for item in palm_ids) / len(palm_ids),
            )
            thumb_tip = points[4]
            index_tip = points[8]
            pinch = Point((thumb_tip.x + index_tip.x) / 2, (thumb_tip.y + index_tip.y) / 2)
            label = "unknown"
            if index < len(handedness) and handedness[index]:
                category = handedness[index][0]
                label = (category.category_name or category.display_name or "unknown").lower()
            observations.append(
                HandObservation(
                    wrist=points[0],
                    palm=palm,
                    pinch=pinch,
                    thumb_tip=thumb_tip,
                    index_tip=index_tip,
                    landmarks=points,
                    handedness=label,
                )
            )
        return observations

    def close(self) -> None:
        self._hands.close()
