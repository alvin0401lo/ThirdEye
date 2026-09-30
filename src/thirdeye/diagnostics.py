from __future__ import annotations

import importlib.util
import platform
import sys

from thirdeye.config import Settings


def run_checks(settings: Settings) -> bool:
    print("ThirdEye AI diagnostics")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"OS: {platform.platform()}")
    required = ["cv2", "mediapipe", "numpy", "openai", "sounddevice", "torch", "ultralytics"]
    if settings.camera_source == "esp32":
        required.extend(["fastapi", "uvicorn", "websockets"])
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    print("Dependencies:", "OK" if not missing else "MISSING " + ", ".join(missing))
    print("OpenAI API key:", "configured" if settings.openai_api_key else "not configured")
    if "torch" not in missing:
        import torch

        print(f"PyTorch: {torch.__version__}; CUDA: {torch.cuda.is_available()}")
    camera_ok = False
    if settings.camera_source == "esp32" and not missing:
        print(f"ESP32 camera endpoint: ws://{settings.camera_host}:{settings.camera_port}/ws/camera")
        from thirdeye.camera import CameraError
        from thirdeye.network_camera import Esp32Camera

        camera = None
        try:
            camera = Esp32Camera(
                settings.camera_host,
                settings.camera_port,
                settings.camera_timeout,
                settings.camera_token,
            )
            frame = camera.read()
            camera_ok = frame is not None
            print("ESP32 camera frame:", "OK" if camera_ok else "unavailable")
        except CameraError as exc:
            print(f"ESP32 camera frame: unavailable ({exc})")
        finally:
            if camera is not None:
                camera.release()
    elif "cv2" not in missing:
        import cv2

        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        capture = cv2.VideoCapture(settings.camera_index, backend)
        camera_ok, frame = capture.read()
        capture.release()
        status = "OK" if camera_ok and frame is not None else "unavailable"
        print(f"Camera index {settings.camera_index}: {status}")
    return not missing and camera_ok
