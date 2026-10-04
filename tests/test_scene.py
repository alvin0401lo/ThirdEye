import numpy as np

from thirdeye.models import Detection, SceneAnalysis, SceneMode
from thirdeye.scene import detection_metadata, render_scene_analysis, select_best_frame


def analysis(**overrides) -> SceneAnalysis:
    values = {
        "immediate_hazards": ("chair directly ahead",),
        "clear_path": "left",
        "people": ("one person on the right",),
        "important_objects": ("table in the center",),
        "doors_and_stairs": ("door on the left",),
        "visible_text": ("EXIT",),
        "uncertainty": (),
    }
    values.update(overrides)
    return SceneAnalysis(**values)


def test_quick_description_prioritizes_hazard_and_path():
    text = render_scene_analysis(analysis(), SceneMode.QUICK)
    assert text.index("Caution") < text.index("path") < text.index("Doors")
    assert "one person" not in text


def test_read_text_mode_only_reads_text():
    assert render_scene_analysis(analysis(), SceneMode.READ_TEXT) == "Visible text: EXIT"


def test_detection_metadata_uses_user_view_regions():
    detections = [
        Detection(0, 0, 100, 100, 0.9, "door"),
        Detection(800, 0, 1000, 100, 0.8, "person"),
    ]
    assert detection_metadata(detections, 1000) == [
        "door at left (90% confidence)",
        "person at right (80% confidence)",
    ]


def test_select_best_frame_prefers_sharper_image():
    flat = np.full((100, 100, 3), 100, dtype=np.uint8)
    checker = np.indices((100, 100)).sum(axis=0) % 2 * 255
    sharp = np.repeat(checker[:, :, None], 3, axis=2).astype(np.uint8)
    selected, quality = select_best_frame([flat, sharp])
    assert selected is sharp
    assert quality.sharpness > 55
