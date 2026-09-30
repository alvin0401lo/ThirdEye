from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from thirdeye.models import Detection, SceneAnalysis, SceneMode


@dataclass(frozen=True, slots=True)
class FrameQuality:
    sharpness: float
    brightness: float
    warnings: tuple[str, ...]


def is_text_question(question: str) -> bool:
    return bool(re.search(
        r"\b(read|text|signs?|labels?|words?|letters?|written|printed|printing|"
        r"prices?|expir(?:y|ation)|receipts?|menus?|numbers?|dates?)\b|"
        r"\bwhat\s+(?:does|do)\s+(?:this|that|it|the\s+\w+)\s+say\b",
        question, re.IGNORECASE,
    ))


def assess_frame(frame: Any) -> FrameQuality:
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    warnings: list[str] = []
    if sharpness < 55:
        warnings.append("image may be blurred")
    if brightness < 45:
        warnings.append("image is dark")
    elif brightness > 220:
        warnings.append("image may be overexposed")
    return FrameQuality(sharpness, brightness, tuple(warnings))


def select_best_frame(frames: list[Any]) -> tuple[Any, FrameQuality]:
    if not frames:
        raise ValueError("At least one frame is required")
    assessed = [(frame, assess_frame(frame)) for frame in frames]
    return max(assessed, key=lambda item: item[1].sharpness)


def detection_metadata(detections: list[Detection], frame_width: int) -> list[str]:
    items: list[str] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True)[:12]:
        ratio = detection.center_x / max(1, frame_width)
        position = "left" if ratio < 0.4 else "right" if ratio > 0.6 else "center"
        items.append(f"{detection.label} at {position} ({detection.confidence:.0%} confidence)")
    return items


def render_scene_analysis(analysis: SceneAnalysis, mode: SceneMode) -> str:
    def joined(items: tuple[str, ...] | list[str]) -> str:
        return "; ".join(item.strip().rstrip(".") for item in items if item.strip())

    if mode is SceneMode.READ_TEXT:
        if analysis.visible_text:
            return "Visible text: " + joined(analysis.visible_text)
        return "I could not read any clear text in this image."

    sentences: list[str] = []
    if analysis.immediate_hazards:
        sentences.append("Caution: " + joined(analysis.immediate_hazards) + ".")
    else:
        sentences.append("No immediate hazard is clearly visible, but the camera view may be incomplete.")

    path = analysis.clear_path.strip().lower()
    if path and path != "unknown":
        sentences.append(f"The clearest visible path appears to be {path}.")
    else:
        sentences.append("A clear path cannot be confirmed from this image.")

    structures = list(analysis.doors_and_stairs)
    if structures:
        sentences.append("Doors or level changes: " + joined(structures) + ".")

    if mode is SceneMode.DETAILED:
        details = list(analysis.people) + list(analysis.important_objects)
        if details:
            sentences.append("Also visible: " + joined(details[:6]) + ".")
        if analysis.visible_text:
            sentences.append("Readable text: " + joined(analysis.visible_text[:3]) + ".")
    if analysis.uncertainty:
        sentences.append("Uncertain: " + joined(analysis.uncertainty[:2]) + ".")
    return " ".join(sentences)
