from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from thirdeye.models import Detection, HandObservation, Point


class FindingMode(str, Enum):
    SEGMENT = "SEGMENT"
    FLASH = "FLASH"
    CENTER_GUIDE = "CENTER_GUIDE"
    TRACK = "TRACK"


class RepoGuidance(str, Enum):
    SEARCHING = "Object not found"
    OBJECT_DETECTED = "Object detected"
    LOCKING = "Locking target"
    CAMERA_LEFT = "Move camera left"
    CAMERA_RIGHT = "Move camera right"
    CAMERA_UP = "Move camera up"
    CAMERA_DOWN = "Move camera down"
    CENTERED = "Object centered"
    SHOW_HAND = "Show your hand"
    LEFT = "Left"
    RIGHT = "Right"
    UP = "Up"
    DOWN = "Down"
    FORWARD = "Forward"
    HOLD = "Hold"


@dataclass(frozen=True, slots=True)
class FindingResult:
    mode: FindingMode
    detection: Detection | None
    hand: HandObservation | None
    guidance: RepoGuidance
    flash_alpha: float = 0.0
    contact_ratio: float = 0.0
    grasp_score: float = 0.0
    range_ratio: float | None = None


class RepoObjectFinder:
    """Object-finding state machine adapted from yolomedia.py in the reference repo."""

    def __init__(
        self,
        detector: Any,
        hand_tracker: Any,
        clock: Callable[[], float] = time.monotonic,
        lock_delay: float = 1.0,
        flash_duration: float = 1.0,
        center_hold: float = 1.0,
        correction_interval: float = 0.35,
    ) -> None:
        self._detector = detector
        self._hand_tracker = hand_tracker
        self._clock = clock
        self._lock_delay = lock_delay
        self._flash_duration = flash_duration
        self._center_hold = center_hold
        self._correction_interval = max(0.0, correction_interval)
        self.reset()

    @property
    def mode(self) -> FindingMode:
        return self._mode

    def set_target(self, target: str) -> None:
        self._detector.set_target(target)
        self.reset()

    def reset(self) -> None:
        self._mode = FindingMode.SEGMENT
        self._detection: Detection | None = None
        self._locked_id: int | None = None
        self._candidate_id: int | None = None
        self._candidate_detection: Detection | None = None
        self._lock_started: float | None = None
        self._flash_started: float | None = None
        self._centered_at: float | None = None
        self._old_gray = None
        self._points = None
        self._next_correction = 0.0
        self._missed_corrections = 0

    def process(self, frame: Any) -> FindingResult:
        now = self._clock()
        hands = self._hand_tracker.track(frame) if self._mode is FindingMode.TRACK else []
        hand = hands[0] if hands else None

        if self._mode is FindingMode.SEGMENT:
            return self._segment(frame, hand, now)
        if self._mode is FindingMode.FLASH:
            return self._flash(frame, hand, now)

        detection = self._track_target(frame)
        if detection is None:
            self.reset()
            return FindingResult(self._mode, None, hand, RepoGuidance.SEARCHING)
        self._detection = detection
        if self._mode is FindingMode.CENTER_GUIDE:
            return self._center_guide(frame, hand, now)
        return self._guide_hand(frame, hand)

    def _segment(self, frame: Any, hand: HandObservation | None, now: float) -> FindingResult:
        detection = select_repo_target(
            self._detector.track(frame, confidence=0.20),
            self._candidate_id,
        )
        if detection is None:
            self._detection = None
            self._candidate_id = None
            self._candidate_detection = None
            self._lock_started = None
            return FindingResult(self._mode, None, hand, RepoGuidance.SEARCHING)

        if not same_repo_candidate(self._candidate_detection, detection, frame.shape):
            self._candidate_detection = detection
            self._candidate_id = detection.track_id
            self._lock_started = now

        self._detection = detection
        if self._lock_started is None:
            self._lock_started = now
        if now - self._lock_started < self._lock_delay:
            return FindingResult(self._mode, detection, hand, RepoGuidance.OBJECT_DETECTED)

        self._locked_id = detection.track_id
        self._mode = FindingMode.FLASH
        self._flash_started = now
        self._lock_started = None
        return FindingResult(self._mode, detection, hand, RepoGuidance.LOCKING)

    def _flash(self, frame: Any, hand: HandObservation | None, now: float) -> FindingResult:
        started = self._flash_started if self._flash_started is not None else now
        elapsed = now - started
        if elapsed < self._flash_duration:
            return FindingResult(
                self._mode,
                self._detection,
                hand,
                RepoGuidance.LOCKING,
                flash_alpha=flash_alpha(elapsed, self._flash_duration),
            )

        # The camera may have moved during the flash; seed from this frame.
        self._detection = select_repo_target(self._detector.track(frame, confidence=0.20), self._locked_id)
        if self._detection is None or not self._seed_flow(frame, self._detection):
            self.reset()
            return FindingResult(self._mode, None, hand, RepoGuidance.SEARCHING)
        self._next_correction = self._clock() + self._correction_interval
        self._mode = FindingMode.CENTER_GUIDE
        self._flash_started = None
        return self._center_guide(frame, hand, now)

    def _center_guide(
        self,
        frame: Any,
        hand: HandObservation | None,
        now: float,
    ) -> FindingResult:
        assert self._detection is not None
        guidance, centered = center_guidance(self._detection, frame.shape)
        if not centered:
            self._centered_at = None
            return FindingResult(self._mode, self._detection, hand, guidance)
        if self._centered_at is None:
            self._centered_at = now
        if now - self._centered_at >= self._center_hold:
            self._mode = FindingMode.TRACK
        return FindingResult(self._mode, self._detection, hand, RepoGuidance.CENTERED)

    def _guide_hand(self, frame: Any, hand: HandObservation | None) -> FindingResult:
        assert self._detection is not None
        if hand is None:
            return FindingResult(self._mode, self._detection, None, RepoGuidance.SHOW_HAND)
        contact_ratio = hand_object_overlap(hand, self._detection, frame.shape)
        grasp_score = repo_grasp_score(hand)
        hand_area = hand_box_area(hand)
        range_ratio = polygon_area(self._detection) / hand_area if hand_area > 0 else None
        guidance = hand_guidance(hand, self._detection, contact_ratio > 0.10)
        return FindingResult(
            self._mode,
            self._detection,
            hand,
            guidance,
            contact_ratio=contact_ratio,
            grasp_score=grasp_score,
            range_ratio=range_ratio,
        )

    def _seed_flow(self, frame: Any, detection: Detection) -> bool:
        import cv2
        import numpy as np

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = detection_mask(detection, frame.shape)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        inner = cv2.erode(mask, kernel, iterations=1)
        edges = cv2.Canny(inner * 255, 50, 150)
        edges = cv2.dilate(
            edges,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=200,
            qualityLevel=0.001,
            minDistance=5,
            blockSize=7,
            mask=edges,
        )
        if points is None or len(points) < 8:
            return False
        self._old_gray = gray
        self._points = np.asarray(points, dtype=np.float32)
        return True

    def _track_target(self, frame: Any) -> Detection | None:
        import cv2
        import numpy as np

        if self._old_gray is None or self._points is None or self._detection is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._old_gray,
            gray,
            self._points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03),
        )
        if next_points is None or status is None:
            return None
        good = next_points[status.reshape(-1) == 1]
        if len(good) < 5:
            return None
        hull = cv2.convexHull(good.reshape(-1, 1, 2)).reshape(-1, 2)
        predicted = detection_from_polygon(hull, self._detection)
        self._old_gray = gray
        self._points = good.reshape(-1, 1, 2)

        if self._clock() < self._next_correction:
            return predicted
        detected = select_repo_target(
            self._detector.track(frame, confidence=0.15),
            self._locked_id,
        )
        # Schedule from completion so slow CPU inference still leaves flow-only frames.
        self._next_correction = self._clock() + self._correction_interval
        if detected is not None and (
            mask_iou(predicted, detected, frame.shape) > 0.20
            or peripheral_match(predicted, detected, frame.shape)
        ):
            if detected.track_id is not None:
                self._locked_id = detected.track_id
            self._missed_corrections = 0
            self._seed_flow(frame, detected)
            return detected
        self._missed_corrections += 1
        # Do not continue guiding to background flow after losing the object.
        return None if self._missed_corrections >= 2 else predicted


def select_repo_target(
    detections: list[Detection],
    locked_id: int | None,
) -> Detection | None:
    if locked_id is not None:
        for detection in detections:
            if detection.track_id == locked_id:
                return detection
    return max(detections, key=polygon_area, default=None)


def same_repo_candidate(
    previous: Detection | None,
    current: Detection,
    shape: tuple[int, ...],
) -> bool:
    if previous is None:
        return False
    if previous.track_id is not None and current.track_id is not None:
        return previous.track_id == current.track_id
    return mask_iou(previous, current, shape) > 0.50


def polygon_area(detection: Detection) -> float:
    import cv2
    import numpy as np

    if len(detection.polygon) < 3:
        return 0.0
    points = np.asarray([(point.x, point.y) for point in detection.polygon], dtype=np.float32)
    return float(cv2.contourArea(points))


def detection_mask(detection: Detection, shape: tuple[int, ...]):
    import cv2
    import numpy as np

    mask = np.zeros(shape[:2], dtype=np.uint8)
    if len(detection.polygon) >= 3:
        polygon = np.asarray(
            [(point.x, point.y) for point in detection.polygon],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [polygon], 1)
    return mask


def detection_from_polygon(points: Any, source: Detection) -> Detection:
    polygon = tuple(Point(float(x), float(y)) for x, y in points)
    return Detection(
        min(point.x for point in polygon),
        min(point.y for point in polygon),
        max(point.x for point in polygon),
        max(point.y for point in polygon),
        source.confidence,
        source.label,
        polygon,
        source.track_id,
    )


def mask_iou(first: Detection, second: Detection, shape: tuple[int, ...]) -> float:
    import numpy as np

    first_mask = detection_mask(first, shape)
    second_mask = detection_mask(second, shape)
    intersection = int(np.logical_and(first_mask, second_mask).sum())
    union = int(np.logical_or(first_mask, second_mask).sum())
    return intersection / union if union else 0.0


def peripheral_match(first: Detection, second: Detection, shape: tuple[int, ...]) -> bool:
    second_mask = detection_mask(second, shape)
    total = int(second_mask.sum())
    if total == 0:
        return False
    height, width = shape[:2]
    x1 = max(0, int(first.x1) - 40)
    y1 = max(0, int(first.y1) - 40)
    x2 = min(width, int(first.x2) + 41)
    y2 = min(height, int(first.y2) + 41)
    return int(second_mask[y1:y2, x1:x2].sum()) > total * 0.10


def center_guidance(
    detection: Detection,
    shape: tuple[int, ...],
    threshold: float = 30.0,
) -> tuple[RepoGuidance, bool]:
    height, width = shape[:2]
    dx = width / 2 - detection.center_x
    dy = height / 2 - detection.center_y
    if (dx * dx + dy * dy) ** 0.5 < threshold:
        return RepoGuidance.CENTERED, True
    if abs(dx) > abs(dy):
        return (RepoGuidance.CAMERA_LEFT if dx > 0 else RepoGuidance.CAMERA_RIGHT), False
    return (RepoGuidance.CAMERA_UP if dy > 0 else RepoGuidance.CAMERA_DOWN), False


def hand_box(hand: HandObservation) -> tuple[int, int, int, int]:
    xs = [point.x for point in hand.landmarks]
    ys = [point.y for point in hand.landmarks]
    x1, y1 = int(min(xs)), int(min(ys))
    return x1, y1, max(1, int(max(xs)) - x1), max(1, int(max(ys)) - y1)


def hand_box_area(hand: HandObservation) -> float:
    _, _, width, height = hand_box(hand)
    return float(width * height)


def hand_center(hand: HandObservation) -> tuple[float, float]:
    return (
        sum(point.x for point in hand.landmarks) / len(hand.landmarks),
        sum(point.y for point in hand.landmarks) / len(hand.landmarks),
    )


def hand_object_overlap(
    hand: HandObservation,
    detection: Detection,
    shape: tuple[int, ...],
) -> float:
    import cv2
    import numpy as np

    height, width = shape[:2]
    x, y, box_width, box_height = hand_box(hand)
    hand_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(
        hand_mask,
        (max(0, x), max(0, y)),
        (min(width - 1, x + box_width), min(height - 1, y + box_height)),
        1,
        -1,
    )
    object_mask = detection_mask(detection, shape)
    intersection = int(np.logical_and(hand_mask, object_mask).sum())
    area = int(hand_mask.sum())
    return intersection / area if area else 0.0


def repo_grasp_score(hand: HandObservation) -> float:
    import numpy as np

    _, _, width, height = hand_box(hand)
    diagonal = float(np.hypot(width, height)) + 1e-6
    thumb = np.asarray((hand.landmarks[4].x, hand.landmarks[4].y))
    index = np.asarray((hand.landmarks[8].x, hand.landmarks[8].y))
    thumb_index = float(np.linalg.norm(thumb - index)) / diagonal
    palm = np.mean(
        np.asarray([(hand.landmarks[i].x, hand.landmarks[i].y) for i in (0, 5, 9, 13, 17)]),
        axis=0,
    )
    curled = sum(
        float(np.linalg.norm(np.asarray((hand.landmarks[i].x, hand.landmarks[i].y)) - palm))
        / diagonal
        < 0.44
        for i in (12, 16, 20)
    )
    return 0.5 * (1.0 - min(thumb_index / 0.34, 1.0)) + 0.5 * min(curled / 3.0, 1.0)


def hand_guidance(
    hand: HandObservation,
    detection: Detection,
    touching: bool,
) -> RepoGuidance:
    if touching:
        return RepoGuidance.FORWARD
    hand_x, hand_y = hand_center(hand)
    dx = detection.center_x - hand_x
    dy = detection.center_y - hand_y
    if abs(dx) > abs(dy) and abs(dx) > 30:
        return RepoGuidance.RIGHT if dx > 0 else RepoGuidance.LEFT
    if abs(dy) > 30:
        return RepoGuidance.DOWN if dy > 0 else RepoGuidance.UP
    if (dx * dx + dy * dy) ** 0.5 < 50:
        return RepoGuidance.FORWARD
    return RepoGuidance.HOLD


def flash_alpha(elapsed: float, duration: float) -> float:
    progress = elapsed / max(duration, 1e-6)
    if progress < 0.3:
        return progress / 0.3 * 0.8
    if progress < 0.7:
        return 0.8
    return max(0.0, (1.0 - progress) / 0.3 * 0.8)
