from pathlib import Path
import shutil
import subprocess

import pytest


def test_firmware_download_retries_before_playback_only(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is needed for the firmware download check")
    root = Path(__file__).resolve().parent.parent
    sketch = (root / "firmware/thirdeye_ai_device/thirdeye_ai_device.ino").read_text(encoding="utf-8")
    u32 = sketch.split("uint32_t wavU32(", 1)[1].split("void reportHttpSpeaker", 1)[0]
    playback = sketch.split("void playHttpSpeaker(void*)", 1)[1].split("CameraPacket copyCameraJpeg", 1)[0]
    harness = Path(__file__).with_name("speaker_download_check.cpp").read_text(encoding="utf-8")
    camera = sketch.split("void onCameraEvent(", 1)[1].split("void onImuEvent", 1)[0]
    source = harness.replace("// INSERT FIRMWARE", "void onCameraEvent(" + camera + "uint32_t wavU32(" + u32 + "void playHttpSpeaker(void*)" + playback)
    executable = tmp_path / "speaker_download_check.exe"
    result = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-pedantic", "-static", "-x", "c++", "-", "-o", str(executable)],
                            input=source, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    subprocess.run([str(executable)], check=True, timeout=10)
