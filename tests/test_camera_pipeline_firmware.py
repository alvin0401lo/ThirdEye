import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
SKETCH = ROOT / "firmware/thirdeye_ai_device/thirdeye_ai_device.ino"


def test_native_heartbeat_enabled_only_for_audio_and_camera():
    sketch = SKETCH.read_text(encoding="utf-8")
    sockets = sketch.split("void connectSockets() {", 1)[1].split("void setup()", 1)[0]
    for name in ("audioSocket", "cameraSocket"):
        assert f"{name}.enableHeartbeat(15000, 5000, 3);" in sockets
        assert sockets.index(f"{name}.begin(") < sockets.index(f"{name}.enableHeartbeat(")
        assert f"{name}.setReconnectInterval(2000);" in sockets
    assert "imuSocket.enableHeartbeat" not in sketch
    assert "LINK_PROBE:" not in sketch


def test_connection_diagnostics_do_not_add_reconnect_calls():
    sketch = SKETCH.read_text(encoding="utf-8")
    assert "Camera heartbeat: %s uptime_ms=%lu" in sketch
    assert "Audio heartbeat: %s uptime_ms=%lu" in sketch
    assert "Wi-Fi disconnected: reason=%u uptime_ms=%lu" in sketch
    assert "Wi-Fi lost IP: uptime_ms=%lu" in sketch
    handler = sketch.split("WiFi.onEvent(", 1)[1].split("WiFi.mode(", 1)[0]
    assert "info.wifi_sta_disconnected.reason" in handler
    assert "delay(" not in handler and "reconnect(" not in handler


def test_camera_mailbox_ownership(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is needed to compile the mailbox ownership check")
    sketch = SKETCH.read_text(encoding="utf-8")
    declarations = sketch.split("struct CameraPacket {", 1)[1].split("CameraMetrics cameraMetrics;", 1)[0]
    mailbox = sketch.split("// BEGIN CAMERA MAILBOX", 1)[1].split("// END CAMERA MAILBOX", 1)[0]
    harness = Path(__file__).with_name("camera_mailbox_check.cpp").read_text(encoding="utf-8")
    source = "#include <cstddef>\n#include <cstdint>\nstruct CameraPacket {" + declarations
    source += harness.replace("// INSERT CAMERA MAILBOX", mailbox)
    executable = tmp_path / "camera_mailbox_check.exe"
    subprocess.run([compiler, "-std=c++11", "-Wall", "-Wextra", "-pedantic", "-static",
                    "-x", "c++", "-", "-o", str(executable)], input=source,
                   check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, timeout=10)


def test_camera_profile_and_single_socket_owner_are_preserved():
    sketch = SKETCH.read_text(encoding="utf-8")
    baseline = (ROOT / "tests/fixtures/thirdeye_device.ino").read_text(encoding="utf-8")
    def camera_config(code):
        return code.split("bool startCamera() {", 1)[1].split("\n}", 1)[0].strip()
    config = camera_config(sketch)
    config = config.replace("  streamFrameSize = hasPsram ? FRAMESIZE_VGA : FRAMESIZE_QVGA;\n", "")
    assert config == camera_config(baseline)
    assert "#define CAMERA_FRAME_INTERVAL_MS 88" in sketch
    assert sketch.count("cameraSocket.loop();") == 1
    loop = sketch.split("void loop() {", 1)[1]
    assert "esp_camera_fb_get" not in loop and "cameraSocket.send" not in loop
    capture = sketch.split("void captureCamera(void*) {", 1)[1].split("void serviceCamera", 1)[0]
    assert "cameraSocket." not in capture
    service = sketch.split("void serviceCamera(void*) {", 1)[1].split("void connectSockets", 1)[0]
    assert "esp_camera_fb_get" not in service and "set_framesize" not in service
