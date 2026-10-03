from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    openai_api_key: str | None
    openai_model: str
    vision_model: str
    transcribe_model: str
    camera_source: str
    audio_source: str
    camera_index: int
    camera_width: int
    camera_height: int
    camera_host: str
    camera_port: int
    camera_timeout: float
    camera_token: str | None
    cue_audio_dir: Path
    detector_model: str
    tracker: str
    confidence: float
    image_size: int
    hand_scale: float
    guidance_repeat_s: float
    language: str
    wake_word: str
    wake_language: str
    wake_confidence: float
    wake_model: str
    microphone_device: int | str | None
    speaker_device: int | str | None
    voice_rate: int
    mirror_hand_view: bool
    companion_bridge_url: str | None = None
    companion_room: str = "THIRDEYE"

    @classmethod
    def load(cls) -> "Settings":
        root = Path(__file__).resolve().parents[2]
        load_dotenv(root / ".env")
        camera_source = os.getenv("THIRDEYE_CAMERA_SOURCE", "webcam").strip().lower()
        audio_source = os.getenv("THIRDEYE_AUDIO_SOURCE", "auto").strip().lower()
        if audio_source not in {"auto", "pc", "esp32"}:
            raise ValueError("THIRDEYE_AUDIO_SOURCE must be 'auto', 'pc', or 'esp32'")
        device_audio = audio_source == "esp32" or (audio_source == "auto" and camera_source == "esp32")
        cue_audio_dir = Path(os.getenv("THIRDEYE_CUE_AUDIO_DIR", root / "audio_cues"))
        if not cue_audio_dir.is_absolute():
            cue_audio_dir = root / cue_audio_dir
        return cls(
            project_root=root,
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_model=os.getenv("THIRDEYE_OPENAI_MODEL", "gpt-5.6-luna"),
            vision_model=os.getenv("THIRDEYE_VISION_MODEL", "gpt-5.6-terra"),
            transcribe_model=os.getenv("THIRDEYE_TRANSCRIBE_MODEL", "local"),
            camera_source=camera_source,
            audio_source=audio_source,
            camera_index=int(os.getenv("THIRDEYE_CAMERA_INDEX", "0")),
            camera_width=int(os.getenv("THIRDEYE_CAMERA_WIDTH", "1280")),
            camera_height=int(os.getenv("THIRDEYE_CAMERA_HEIGHT", "720")),
            camera_host=os.getenv("THIRDEYE_CAMERA_HOST", "0.0.0.0"),
            camera_port=int(os.getenv("THIRDEYE_CAMERA_PORT", "8081")),
            camera_timeout=float(os.getenv("THIRDEYE_CAMERA_TIMEOUT", "5")),
            camera_token=os.getenv("THIRDEYE_CAMERA_TOKEN") or None,
            cue_audio_dir=cue_audio_dir,
            detector_model=os.getenv("THIRDEYE_MODEL_PATH", "yoloe-26s-seg.pt"),
            tracker=os.getenv("THIRDEYE_TRACKER", "bytetrack.yaml"),
            confidence=float(os.getenv("THIRDEYE_CONFIDENCE", "0.20")),
            image_size=int(os.getenv("THIRDEYE_IMAGE_SIZE", "640")),
            hand_scale=max(0.25, min(1.0, float(os.getenv("THIRDEYE_HAND_SCALE", "0.8")))),
            guidance_repeat_s=float(os.getenv("THIRDEYE_GUIDANCE_REPEAT_S", "1.5")),
            language=os.getenv("THIRDEYE_LANGUAGE", "en"),
            wake_word=os.getenv("THIRDEYE_WAKE_WORD", "hi third eye"),
            wake_language=os.getenv("THIRDEYE_WAKE_LANGUAGE", "en-US"),
            wake_confidence=float(os.getenv("THIRDEYE_WAKE_CONFIDENCE", "0.70")),
            wake_model=os.getenv("THIRDEYE_WAKE_MODEL", "small.en"),
            microphone_device=None if device_audio else _parse_audio_device(os.getenv("THIRDEYE_MICROPHONE_DEVICE")),
            speaker_device=None if device_audio else _parse_audio_device(os.getenv("THIRDEYE_SPEAKER_DEVICE")),
            voice_rate=int(os.getenv("THIRDEYE_VOICE_RATE", "0")),
            mirror_hand_view=os.getenv("THIRDEYE_MIRROR_HAND_VIEW", "true").lower() in {"1", "true", "yes", "on"},
            companion_bridge_url=os.getenv("THIRDEYE_COMPANION_BRIDGE_URL", "ws://127.0.0.1:4174").strip() or None,
            companion_room=os.getenv("THIRDEYE_COMPANION_ROOM", "THIRDEYE").strip().upper() or "THIRDEYE",
        )


def _parse_audio_device(value: str | None) -> int | str | None:
    if not value or not value.strip():
        return None
    cleaned = value.strip()
    if cleaned.lower() == "esp32":
        raise ValueError(
            "'esp32' is an audio source, not a PC device name. "
            "Set THIRDEYE_AUDIO_SOURCE=esp32 for board audio, or leave "
            "THIRDEYE_MICROPHONE_DEVICE and THIRDEYE_SPEAKER_DEVICE blank for default PC audio. "
            "PowerShell environment variables override .env."
        )
    try:
        return int(cleaned)
    except ValueError:
        return cleaned
