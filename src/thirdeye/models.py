from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Intent(str, Enum):
    FIND_OBJECT = "FIND_OBJECT"
    DESCRIBE_SCENE = "DESCRIBE_SCENE"
    VISUAL_QUESTION = "VISUAL_QUESTION"
    STANDBY = "STANDBY"
    EXIT = "EXIT"
    UNKNOWN = "UNKNOWN"


class SceneMode(str, Enum):
    QUICK = "quick"
    DETAILED = "detailed"
    READ_TEXT = "read_text"


@dataclass(frozen=True, slots=True)
class Command:
    intent: Intent
    target: str | None = None


@dataclass(frozen=True, slots=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str
    polygon: tuple[Point, ...] = ()
    track_id: int | None = None

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class HandObservation:
    wrist: Point
    palm: Point
    pinch: Point
    thumb_tip: Point
    index_tip: Point
    landmarks: tuple[Point, ...]
    handedness: str = "unknown"


@dataclass(frozen=True, slots=True)
class FramePacket:
    frame_id: int
    timestamp: float
    image: Any
    source: str


@dataclass(frozen=True, slots=True)
class SceneAnalysis:
    immediate_hazards: tuple[str, ...]
    clear_path: str
    people: tuple[str, ...]
    important_objects: tuple[str, ...]
    doors_and_stairs: tuple[str, ...]
    visible_text: tuple[str, ...]
    uncertainty: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VisualAnswer:
    answer: str
    can_answer: bool
    evidence: str
    subject: str = ""
