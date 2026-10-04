from __future__ import annotations

import base64
import json
import re
from typing import Any

from thirdeye.models import Command, Detection, Intent, SceneAnalysis, SceneMode, VisualAnswer
from thirdeye.scene import FrameQuality, detection_metadata, is_text_question


class AIUnavailable(RuntimeError):
    pass


TARGET_TRANSLATIONS = {
    "水瓶": "water bottle",
    "瓶子": "bottle",
    "杯子": "cup",
    "手机": "cell phone",
    "电话": "cell phone",
    "钥匙": "keys",
    "钱包": "wallet",
    "眼镜": "glasses",
    "书": "book",
    "椅子": "chair",
    "背包": "backpack",
    "遥控器": "remote control",
    "药瓶": "medicine bottle",
}


def normalize_visual_target(target: str) -> str:
    cleaned = " ".join(target.strip().lower().split()).strip(" .,!?。！？")
    compact = cleaned.replace("我的", "").replace("一个", "").strip()
    return TARGET_TRANSLATIONS.get(compact, cleaned)


def is_english_command_text(text: str) -> bool:
    """Accept simple English commands and reject non-ASCII transcription output."""
    return text.isascii() and any(character.isalpha() for character in text)


def _local_command(text: str) -> Command:
    cleaned = " ".join(text.strip().split())
    lowered = cleaned.lower()
    if len(re.findall(r"[.!?]", cleaned)) >= 4:
        return Command(Intent.UNKNOWN)
    if lowered in {"shutdown third eye", "close third eye"}:
        return Command(Intent.EXIT)
    if lowered in {"exit", "quit", "stop", "退出", "结束"}:
        return Command(Intent.STANDBY)
    if any(phrase in lowered for phrase in ("describe scene", "describe the scene", "scene description", "描述场景", "描述环境")):
        return Command(Intent.DESCRIBE_SCENE)
    patterns = (
        r"(?:help me )?find (?:my |the |a |an )?(.+?)[.!?]?$",
        r"(?:look for|locate) (?:my |the |a |an )?(.+?)[.!?]?$",
        r"(?:帮我)?找(?:一下)?(.+?)[。！？]?$",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            target = match.group(1).strip(" .,!?")
            if target in {"a", "an", "the", "my"} or "..." in match.group(0):
                return Command(Intent.FIND_OBJECT)
            return Command(Intent.FIND_OBJECT, normalize_visual_target(target) if target else None)
    if any(phrase in lowered for phrase in (
        "what do you see", "what can you see", "what is in front", "what's in front",
        "what is ahead", "what's ahead", "what's around me", "what is around me",
        "what am i holding", "what's in my hand", "what is in my hand",
        "what does the sign say", "what does this sign say", "what does this label say",
        "read the sign", "read the text", "read this sign", "read this label",
        "what kind of area", "which aisle", "which section", "what section",
        "what is on my left", "what's on my left", "what is on my right", "what's on my right",
        "what color is it", "what colour is it", "what does it say", "where is it",
        "can you read it", "what is it made of",
    )) or lowered.startswith(("is it ", "is that ")) or re.search(
        r"\b(?:is there (?:a|an)|are there any) (?:door|person|people|stair|stairs|sign|obstacle|chair|table|car|shelf|bottle)\b",
        lowered,
    ):
        return Command(Intent.VISUAL_QUESTION)
    return Command(Intent.UNKNOWN)


class AIService:
    def __init__(
        self,
        api_key: str | None,
        model: str,
        transcribe_model: str,
        vision_model: str | None = None,
        language: str = "en",
    ) -> None:
        self._model = model
        self._vision_model = vision_model or model
        self._transcribe_model = transcribe_model
        self._language = language
        self._client: Any | None = None
        if api_key:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key)

    @property
    def online(self) -> bool:
        return self._client is not None

    def understand_command(self, text: str) -> Command:
        local = _local_command(text)
        if len(re.findall(r"[.!?]", text)) >= 4:
            return local
        target_is_english = local.target is None or local.target.isascii()
        if (
            local.intent is not Intent.UNKNOWN
            and (local.intent is not Intent.FIND_OBJECT or target_is_english)
        ) or not self._client:
            return local
        schema = {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": [item.value for item in Intent]},
                "target": {"type": ["string", "null"]},
            },
            "required": ["intent", "target"],
            "additionalProperties": False,
        }
        response = self._client.responses.create(
            model=self._model,
            instructions=(
                "Classify a command for a visual assistance app. For FIND_OBJECT, return a concise "
                "English visual category in target. Use STANDBY to end the current interaction, "
                "and EXIT only when the user explicitly asks to shut down ThirdEye. Use "
                "DESCRIBE_SCENE for a general navigation overview and VISUAL_QUESTION for "
                "questions requiring the current camera view (objects, signs, text, surroundings). "
                "Use UNKNOWN for questions unrelated to the camera. Use null for all targets "
                "except FIND_OBJECT."
            ),
            input=text,
            text={"format": {"type": "json_schema", "name": "command", "schema": schema, "strict": True}},
        )
        data = json.loads(response.output_text)
        target = data.get("target")
        return Command(Intent(data["intent"]), target.strip().lower() if target else None)

    def transcribe(self, audio_path: str) -> str:
        if not self._client:
            raise AIUnavailable("OPENAI_API_KEY is not configured")
        with open(audio_path, "rb") as audio_file:
            result = self._client.audio.transcriptions.create(
                model=self._transcribe_model,
                file=audio_file,
                language=self._language,
            )
        return result.text.strip()

    def answer_visual_question(
        self,
        frame: Any,
        question: str,
        quality: FrameQuality | None = None,
    ) -> VisualAnswer:
        if not self._client:
            raise AIUnavailable("OPENAI_API_KEY is required for visual questions")
        import cv2

        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95 if is_text_question(question) else 85])
        if not ok:
            raise RuntimeError("Could not encode camera frame")
        image_url = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "can_answer": {"type": "boolean"},
                "evidence": {"type": "string"},
                "subject": {"type": "string"},
            },
            "required": ["answer", "can_answer", "evidence", "subject"],
            "additionalProperties": False,
        }
        warnings = list(quality.warnings) if quality else []
        response = self._client.responses.create(
            model=self._vision_model,
            instructions=(
                "Answer the user's visual question using only the current forward-facing camera image. "
                "Reply in concise English with one short spoken sentence, ideally no more than 12 words. "
                "For text-reading questions, preserve the exact readable text even when longer. "
                "Do not replace the answer "
                "with a generic scene or navigation summary. Report exact printed text only when legible; "
                "do not guess obscured objects, store aisles, signs, distances, or a safe route. "
                "For an inferred area, say 'appears to be' and name visible evidence. "
                "If the requested detail is not visible or the image is too poor, set can_answer "
                "to false and answer with a brief reason or request to adjust the camera. "
                "Set subject to a short visible noun phrase only when the answer singles out one "
                "specific physical object, such as 'the bottle in the hand'. Use 'the' rather than 'a'. "
                "Otherwise use an empty string. "
                "Never claim a path is safe."
            ),
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"Question: {question}\nImage quality warnings: {warnings or ['none']}"},
                    {
                        "type": "input_image",
                        "image_url": image_url,
                        "detail": "high" if is_text_question(question) else "low",
                    },
                ],
            }],
            text={"format": {"type": "json_schema", "name": "visual_answer", "schema": schema, "strict": True}},
        )
        data = json.loads(response.output_text)
        return VisualAnswer(
            data["answer"].strip(), data["can_answer"], data["evidence"].strip(),
            data["subject"].strip(),
        )

    def describe_scene(
        self,
        frame: Any,
        mode: SceneMode = SceneMode.QUICK,
        question: str | None = None,
        local_detections: list[Detection] | None = None,
        quality: FrameQuality | None = None,
    ) -> SceneAnalysis:
        if not self._client:
            raise AIUnavailable("OPENAI_API_KEY is required for scene description")
        import cv2

        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95 if mode is SceneMode.READ_TEXT else 85])
        if not ok:
            raise RuntimeError("Could not encode camera frame")
        image_url = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
        schema = {
            "type": "object",
            "properties": {
                "immediate_hazards": {"type": "array", "items": {"type": "string"}},
                "clear_path": {
                    "type": "string",
                    "enum": ["left", "center", "right", "multiple", "blocked", "unknown"],
                },
                "people": {"type": "array", "items": {"type": "string"}},
                "important_objects": {"type": "array", "items": {"type": "string"}},
                "doors_and_stairs": {"type": "array", "items": {"type": "string"}},
                "visible_text": {"type": "array", "items": {"type": "string"}},
                "uncertainty": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "immediate_hazards",
                "clear_path",
                "people",
                "important_objects",
                "doors_and_stairs",
                "visible_text",
                "uncertainty",
            ],
            "additionalProperties": False,
        }
        metadata = detection_metadata(local_detections or [], frame.shape[1])
        quality_notes = list(quality.warnings) if quality else []
        user_request = question or {
            SceneMode.QUICK: "Give a quick navigation-oriented scene overview.",
            SceneMode.DETAILED: "Describe the useful scene details for orientation.",
            SceneMode.READ_TEXT: "Read all clearly visible printed text exactly and in natural order.",
        }[mode]
        prompt = (
            f"User request: {user_request}\n"
            f"Local detector observations (may be wrong): {metadata or ['none']}\n"
            f"Camera quality warnings: {quality_notes or ['none']}"
        )
        response = self._client.responses.create(
            model=self._vision_model,
            instructions=(
                "Analyze a first-person camera image for a blind or low-vision user. Prioritize immediate obstacles, "
                "drop-offs, stairs, doors, people, moving hazards, and the clearest visible path. Use the user's "
                "left, center, and right. Report only visible evidence, keep uncertain claims in uncertainty, and never "
                "claim that a route is safe. Transcribe text only when legible. Each visible_text item must contain only "
                "the exact printed text, without location or commentary. Treat local detections as hints, not facts."
            ),
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": image_url,
                        "detail": "low" if mode is SceneMode.QUICK else "high",
                    },
                ],
            }],
            reasoning={"effort": "low" if mode is SceneMode.QUICK else "medium"},
            text={"format": {"type": "json_schema", "name": "scene_analysis", "schema": schema, "strict": True}},
        )
        data = json.loads(response.output_text)
        return SceneAnalysis(
            immediate_hazards=tuple(data["immediate_hazards"]),
            clear_path=data["clear_path"],
            people=tuple(data["people"]),
            important_objects=tuple(data["important_objects"]),
            doors_and_stairs=tuple(data["doors_and_stairs"]),
            visible_text=tuple(data["visible_text"]),
            uncertainty=tuple(data["uncertainty"]),
        )
