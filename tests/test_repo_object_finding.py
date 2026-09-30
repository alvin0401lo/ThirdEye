import numpy as np

from thirdeye.models import Detection, HandObservation, Point
from thirdeye.repo_object_finding import (
    FindingMode,
    RepoGuidance,
    RepoObjectFinder,
    center_guidance,
    hand_guidance,
    hand_object_overlap,
    select_repo_target,
)


def _detection(x1=40, y1=40, x2=60, y2=60, track_id=None):
    return Detection(
        x1,
        y1,
        x2,
        y2,
        0.9,
        "bottle",
        (Point(x1, y1), Point(x2, y1), Point(x2, y2), Point(x1, y2)),
        track_id,
    )


def _hand(x1=10, y1=40, x2=30, y2=60):
    points = [Point(x1, y1), Point(x2, y2)]
    points.extend(Point((x1 + x2) / 2, (y1 + y2) / 2) for _ in range(19))
    landmarks = tuple(points[:21])
    return HandObservation(
        wrist=landmarks[0],
        palm=landmarks[9],
        pinch=landmarks[8],
        thumb_tip=landmarks[4],
        index_tip=landmarks[8],
        landmarks=landmarks,
    )


class _Detector:
    def __init__(self, detections):
        self.detections = detections
        self.target = None
        self.calls = 0

    def set_target(self, target):
        self.target = target

    def track(self, _frame, **_kwargs):
        self.calls += 1
        return list(self.detections)


class _SequenceDetector:
    def __init__(self, frames):
        self.frames = list(frames)
        self.target = None
        self.calls = 0

    def set_target(self, target):
        self.target = target

    def track(self, _frame, **_kwargs):
        self.calls += 1
        if not self.frames:
            return []
        return list(self.frames.pop(0))


class _Hands:
    def __init__(self, hands=()):
        self.hands = list(hands)

    def track(self, _frame):
        return list(self.hands)


def test_repo_target_prefers_locked_id_then_largest_mask():
    small = _detection(0, 0, 10, 10, track_id=5)
    large = _detection(0, 0, 30, 30, track_id=8)

    assert select_repo_target([small, large], locked_id=5) is small
    assert select_repo_target([small, large], locked_id=None) is large


def test_repo_state_machine_segment_flash_center_track():
    now = [0.0]
    detection = _detection(track_id=7)
    finder = RepoObjectFinder(
        _Detector([detection]),
        _Hands(),
        clock=lambda: now[0],
    )
    finder.set_target("bottle")
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    assert finder.process(frame).mode is FindingMode.SEGMENT
    now[0] = 1.0
    assert finder.process(frame).mode is FindingMode.FLASH
    finder._seed_flow = lambda *_args: True
    now[0] = 2.0
    assert finder.process(frame).mode is FindingMode.CENTER_GUIDE
    finder._track_target = lambda *_args: detection
    now[0] = 3.0
    assert finder.process(frame).mode is FindingMode.TRACK


def test_segment_lock_timer_resets_when_candidate_changes():
    now = [0.0]
    first = _detection(0, 0, 20, 20, track_id=1)
    second = _detection(70, 70, 95, 95, track_id=2)
    finder = RepoObjectFinder(
        _SequenceDetector([[first], [second], [second]]),
        _Hands(),
        clock=lambda: now[0],
    )
    finder.set_target("bottle")
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    assert finder.process(frame).mode is FindingMode.SEGMENT
    now[0] = 1.0
    assert finder.process(frame).mode is FindingMode.SEGMENT
    now[0] = 2.0
    assert finder.process(frame).mode is FindingMode.FLASH


def test_center_guidance_matches_reference_camera_directions():
    guidance, centered = center_guidance(_detection(0, 40, 20, 60), (100, 100, 3))
    assert guidance is RepoGuidance.CAMERA_LEFT
    assert not centered

    guidance, centered = center_guidance(_detection(), (100, 100, 3))
    assert guidance is RepoGuidance.CENTERED
    assert centered


def test_hand_guidance_uses_hand_center_and_contact_override():
    hand = _hand()
    detection = _detection(60, 40, 80, 60)

    assert hand_guidance(hand, detection, touching=False) is RepoGuidance.RIGHT
    assert hand_guidance(hand, detection, touching=True) is RepoGuidance.FORWARD


def test_contact_is_hand_rectangle_to_object_mask_overlap():
    hand = _hand(40, 40, 60, 60)
    detection = _detection(50, 50, 70, 70)

    assert hand_object_overlap(hand, detection, (100, 100, 3)) > 0.10


def test_track_mode_runs_yoloe_each_frame_for_correction():
    detection = _detection(track_id=7)
    detector = _Detector([detection])
    finder = RepoObjectFinder(detector, _Hands())
    finder._mode = FindingMode.TRACK
    finder._detection = detection
    finder._track_target = lambda frame: select_repo_target(detector.track(frame), 7)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    finder.process(frame)
    finder.process(frame)

    assert detector.calls == 2
