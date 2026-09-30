import time
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from thirdeye.app import ThirdEyeApp
from thirdeye.camera import CameraError
from thirdeye.models import FramePacket, SceneMode, VisualAnswer
from thirdeye.scene import FrameQuality


class FakeWorker:
    def __init__(self, packets):
        self.packets = list(packets)

    def read(self, after_frame_id=-1, timeout=5.0):
        if not self.packets:
            raise CameraError("No new frame")
        packet = self.packets.pop(0)
        assert packet.frame_id > after_frame_id
        return packet


def test_visual_capture_waits_for_frame_after_question():
    now = time.monotonic()
    stale = FramePacket(1, now - 10, np.zeros((8, 8, 3), dtype=np.uint8), "esp32")
    fresh = FramePacket(2, now, np.full((8, 8, 3), 100, dtype=np.uint8), "esp32")
    app = object.__new__(ThirdEyeApp)
    app.settings = SimpleNamespace(camera_timeout=1)
    app._start_camera_worker = lambda: FakeWorker([stale, fresh])

    frame, _ = app._capture_scene_frame(count=1)

    assert np.array_equal(frame, fresh.image)


def test_visual_capture_rejects_stale_frame_when_camera_stops():
    stale = FramePacket(1, time.monotonic() - 10, np.zeros((8, 8, 3), dtype=np.uint8), "esp32")
    app = object.__new__(ThirdEyeApp)
    app.settings = SimpleNamespace(camera_timeout=1)
    app._start_camera_worker = lambda: FakeWorker([stale])

    with pytest.raises(CameraError, match="No fresh camera frame"):
        app._capture_scene_frame(count=1)


def test_visual_answer_does_not_speak_uncertain_guess():
    spoken = []
    app = object.__new__(ThirdEyeApp)
    app.ai = SimpleNamespace(
        online=True,
        answer_visual_question=lambda frame, question, quality: VisualAnswer(
            "This is definitely the drinks aisle.", False, ""
        ),
    )
    app._capture_scene_frame = lambda count: (
        np.zeros((8, 8, 3), dtype=np.uint8), FrameQuality(100, 100, ()),
    )
    app._speak_to_user = spoken.append

    app.answer_visual_question("Which aisle is this?")

    assert spoken == ["I cannot tell from this camera view. Please adjust the camera and try again."]


@pytest.mark.parametrize(
    ("question", "expected_count"),
    [("What is in my hand?", 2), ("Read this label", 5),
     ("What does it say?", 5), ("What does this say?", 5),
     ("What is the expiry date?", 5)],
)
def test_visual_questions_capture_frames_by_task(question, expected_count):
    counts = []
    app = object.__new__(ThirdEyeApp)
    app._visual_context = None
    app.ai = SimpleNamespace(
        online=True,
        answer_visual_question=lambda frame, text, quality: VisualAnswer("Visible.", True, "Visible"),
    )
    app._capture_scene_frame = lambda count: (
        counts.append(count) or np.zeros((80, 80, 3), dtype=np.uint8),
        FrameQuality(100, 100, ()),
    )
    app._capture_text_frame = lambda: (
        counts.append(5) or np.zeros((80, 80, 3), dtype=np.uint8),
        FrameQuality(100, 100, ()),
    )
    app._speak_to_user = lambda text: None

    app.answer_visual_question(question)

    assert counts == [expected_count]


@pytest.mark.parametrize(
    ("mode", "expected_count"),
    [(SceneMode.QUICK, 2), (SceneMode.DETAILED, 3), (SceneMode.READ_TEXT, 5)],
)
def test_scene_modes_capture_frames_by_task(mode, expected_count, monkeypatch):
    counts = []
    app = object.__new__(ThirdEyeApp)
    app.ai = SimpleNamespace(online=True, describe_scene=lambda *args, **kwargs: object())
    app._capture_scene_frame = lambda count: (
        counts.append(count) or np.zeros((80, 80, 3), dtype=np.uint8),
        FrameQuality(100, 100, ()),
    )
    app._capture_text_frame = lambda: (
        counts.append(5) or np.zeros((80, 80, 3), dtype=np.uint8),
        FrameQuality(100, 100, ()),
    )
    app._load_detector = lambda: SimpleNamespace(set_classes=lambda classes: None, detect=lambda frame: [])
    app._speak_to_user = lambda text: None
    monkeypatch.setattr("thirdeye.app.render_scene_analysis", lambda analysis, scene_mode: "Scene")
    monkeypatch.setattr("cv2.imshow", lambda *args: None)
    monkeypatch.setattr("cv2.waitKey", lambda *args: None)
    monkeypatch.setattr("cv2.destroyAllWindows", lambda: None)

    app.describe_scene(mode)

    assert counts == [expected_count]


def test_text_capture_uses_esp32_still_without_replacing_stream():
    app = object.__new__(ThirdEyeApp)
    app.settings = SimpleNamespace(camera_source="esp32", camera_timeout=2, mirror_hand_view=False)
    photo = np.zeros((1200, 1600, 3), dtype=np.uint8)
    calls = []
    video_changes = []

    @contextmanager
    def video_session():
        video_changes.append("ON")
        try:
            yield
        finally:
            video_changes.append("OFF")

    app._network_camera = SimpleNamespace(
        capture_still=lambda timeout: calls.append(timeout) or photo, video_session=video_session,
    )
    app._start_camera_worker = lambda: None

    frame, _ = app._capture_text_frame()

    assert frame is photo
    assert calls == [8.0]
    assert video_changes == ["ON", "OFF"]


def test_visual_followup_resolves_it_without_sending_old_image():
    spoken = []
    requests = []
    frames = [
        np.full((80, 80, 3), 80, dtype=np.uint8),
        np.full((80, 80, 3), 81, dtype=np.uint8),
    ]
    app = object.__new__(ThirdEyeApp)
    app._visual_context = None
    app.ai = SimpleNamespace(
        online=True,
        answer_visual_question=lambda frame, question, quality: (
            requests.append((frame, question))
            or VisualAnswer("A bottle.", True, "A bottle is visible", "the bottle in my hand")
        ),
    )
    app._capture_scene_frame = lambda count: (frames.pop(0), FrameQuality(100, 100, ()))
    app._speak_to_user = spoken.append

    app.answer_visual_question("What is in my hand?")
    app.answer_visual_question("What color is it?")

    assert [question for _, question in requests] == [
        "What is in my hand?", "What color is the bottle in my hand?"
    ]
    assert requests[0][0] is not requests[1][0]
    assert len(spoken) == 2


def test_visual_followup_forgets_subject_after_scene_change():
    spoken = []
    requests = []
    frames = [
        np.zeros((80, 80, 3), dtype=np.uint8),
        np.full((80, 80, 3), 255, dtype=np.uint8),
    ]
    app = object.__new__(ThirdEyeApp)
    app._visual_context = None
    app.ai = SimpleNamespace(
        online=True,
        answer_visual_question=lambda frame, question, quality: (
            requests.append(question) or VisualAnswer("A bottle.", True, "A bottle is visible", "the bottle")
        ),
    )
    app._capture_scene_frame = lambda count: (frames.pop(0), FrameQuality(100, 100, ()))
    app._speak_to_user = spoken.append

    app.answer_visual_question("What is in my hand?")
    app.answer_visual_question("What color is it?")

    assert requests == ["What is in my hand?"]
    assert spoken[-1] == "I lost track of the object. Please ask the full question again."


def test_visual_followup_expires_and_requires_a_clear_subject():
    app = object.__new__(ThirdEyeApp)
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    app._visual_context = None
    assert app._resolve_visual_followup("What color is it?", frame) is None
    app._visual_context = ("the bottle", time.monotonic() - 16, app._visual_thumbnail(frame))
    assert app._resolve_visual_followup("What color is it?", frame) is None
