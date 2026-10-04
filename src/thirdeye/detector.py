from __future__ import annotations

from typing import Any

from thirdeye.models import Detection, Point


TARGET_PROMPTS = {
    "water bottle": ("water bottle", "bottle", "plastic bottle"),
    "bottle": ("bottle", "water bottle", "plastic bottle"),
    "cell phone": ("cell phone", "smartphone", "mobile phone"),
    "keys": ("keys", "keyring", "car keys"),
    "wallet": ("wallet", "purse"),
    "glasses": ("eyeglasses", "glasses", "spectacles"),
    "remote control": ("remote control", "TV remote"),
    "medicine bottle": ("medicine bottle", "pill bottle"),
}


def target_prompts(target: str) -> list[str]:
    normalized = " ".join(target.strip().lower().split())
    return list(TARGET_PROMPTS.get(normalized, (normalized,)))


def _clean_polygon(points: Any, frame_shape: tuple[int, ...]) -> tuple[Point, ...]:
    """Keep the main connected mask and remove single-frame segmentation speckle."""
    import cv2
    import numpy as np

    if points is None or len(points) < 3:
        return ()
    height, width = frame_shape[:2]
    polygon = np.asarray(points, dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ()
    contour = max(contours, key=cv2.contourArea)
    minimum_area = max(100.0, height * width * 0.0002)
    if cv2.contourArea(contour) < minimum_area:
        return ()
    epsilon = 0.002 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return tuple(Point(float(x), float(y)) for x, y in contour)


class ObjectDetector:
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        image_size: int = 480,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        from ultralytics import YOLOE

        self._model = YOLOE(model_path)
        self._confidence = confidence
        self._image_size = image_size
        self._tracker = tracker
        self._classes: tuple[str, ...] = ()
        self._target: str | None = None

    @property
    def target(self) -> str | None:
        return self._target

    def set_target(self, target: str) -> None:
        self._target = " ".join(target.strip().lower().split())
        self.set_classes(target_prompts(self._target))
        self._reset_tracking()

    def set_classes(self, classes: list[str]) -> None:
        normalized = tuple(" ".join(item.strip().lower().split()) for item in classes if item.strip())
        if not normalized:
            raise ValueError("At least one class is required")
        if normalized != self._classes:
            self._model.set_classes(list(normalized))
            self._classes = normalized
            self._reset_tracking()

    def detect(self, frame: Any) -> list[Detection]:
        return self._infer(frame, tracking=False)

    def track(self, frame: Any, confidence: float | None = None) -> list[Detection]:
        return self._infer(frame, tracking=True, confidence=confidence)

    def _infer(
        self,
        frame: Any,
        tracking: bool,
        confidence: float | None = None,
    ) -> list[Detection]:
        if not self._classes:
            raise RuntimeError("Set one or more classes before running detection")
        options = {
            "conf": self._confidence if confidence is None else confidence,
            "imgsz": self._image_size,
            "iou": 0.45,
            "agnostic_nms": True,
            "verbose": False,
        }
        if tracking:
            result = self._model.track(
                frame,
                persist=True,
                tracker=self._tracker,
                **options,
            )[0]
        else:
            result = self._model.predict(frame, **options)[0]
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        mask_polygons = result.masks.xy if result.masks is not None else []
        for index, box in enumerate(result.boxes):
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            box_id = getattr(box, "id", None)
            track_id = int(box_id[0]) if box_id is not None else None
            fallback = self._classes[min(class_id, len(self._classes) - 1)]
            label = result.names.get(class_id, fallback) if isinstance(result.names, dict) else result.names[class_id]
            polygon = ()
            if index < len(mask_polygons):
                polygon = _clean_polygon(mask_polygons[index], frame.shape)
            if not polygon:
                continue
            x1 = min(point.x for point in polygon)
            y1 = min(point.y for point in polygon)
            x2 = max(point.x for point in polygon)
            y2 = max(point.y for point in polygon)
            detections.append(
                Detection(x1, y1, x2, y2, confidence, str(label), polygon, track_id)
            )
        return detections

    def _reset_tracking(self) -> None:
        predictor = getattr(self._model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []) or []:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()
