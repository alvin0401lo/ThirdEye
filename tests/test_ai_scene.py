import json
from types import SimpleNamespace

import numpy as np
import pytest

from thirdeye.ai_service import AIService
from thirdeye.models import SceneMode
from thirdeye.scene import assess_frame


class FakeResponses:
    def __init__(self, output=None) -> None:
        self.request = None
        self.output = output or {
            "immediate_hazards": ["chair ahead"],
            "clear_path": "left",
            "people": [],
            "important_objects": ["table in center"],
            "doors_and_stairs": ["door on left"],
            "visible_text": ["EXIT"],
            "uncertainty": [],
        }

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(output_text=json.dumps(self.output))


def test_scene_request_uses_structured_output_and_high_detail():
    service = AIService(None, "command-model", "transcribe-model", "vision-model")
    responses = FakeResponses()
    service._client = SimpleNamespace(responses=responses)
    frame = np.full((120, 160, 3), 100, dtype=np.uint8)

    result = service.describe_scene(
        frame,
        mode=SceneMode.DETAILED,
        question="Where is the exit?",
        quality=assess_frame(frame),
    )

    assert result.clear_path == "left"
    assert result.visible_text == ("EXIT",)
    assert responses.request["model"] == "vision-model"
    assert responses.request["text"]["format"]["type"] == "json_schema"
    image_part = responses.request["input"][0]["content"][1]
    assert image_part["detail"] == "high"


def test_visual_question_sends_one_image_and_original_question():
    service = AIService(None, "command-model", "transcribe-model", "vision-model")
    responses = FakeResponses({
        "answer": "The sign says Drinks.",
        "can_answer": True,
        "evidence": "Drinks is legible on the sign",
        "subject": "the Drinks sign",
    })
    service._client = SimpleNamespace(responses=responses)

    result = service.answer_visual_question(
        np.full((120, 160, 3), 100, dtype=np.uint8), "What does the sign say?"
    )

    content = responses.request["input"][0]["content"]
    assert content[0]["text"].startswith("Question: What does the sign say?")
    assert len(content) == 2
    assert content[1]["detail"] == "high"
    assert responses.request["model"] == "vision-model"
    assert result.answer == "The sign says Drinks."
    assert result.subject == "the Drinks sign"


def test_visual_question_uses_low_detail_for_non_text_question():
    service = AIService(None, "command-model", "transcribe-model", "vision-model")
    service._client = SimpleNamespace(responses=FakeResponses({
        "answer": "A red mug is on the table.", "can_answer": True,
        "evidence": "The mug is visibly red.", "subject": "the mug",
    }))

    service.answer_visual_question(
        np.full((120, 160, 3), 100, dtype=np.uint8), "What color is the mug?"
    )

    content = service._client.responses.request["input"][0]["content"]
    assert content[1]["detail"] == "low"


@pytest.mark.parametrize("question,expected_quality", [
    ("What does this label say?", 95),
    ("What does this say?", 95),
    ("What color is it?", 85),
])
def test_text_question_preserves_jpeg_detail(monkeypatch, question, expected_quality):
    import cv2

    service = AIService(None, "command-model", "transcribe-model", "vision-model")
    service._client = SimpleNamespace(responses=FakeResponses({
        "answer": "Cannot tell", "can_answer": False, "evidence": "", "subject": "",
    }))
    original = cv2.imencode
    used_quality = []

    def encode(extension, image, options):
        used_quality.append(options[1])
        return original(extension, image, options)

    monkeypatch.setattr(cv2, "imencode", encode)
    service.answer_visual_question(np.full((120, 160, 3), 100, dtype=np.uint8), question)
    assert used_quality == [expected_quality]
