import json
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest
from websockets.sync.client import connect

from thirdeye.network_camera import Esp32Camera


def test_firmware_fall_state_machine(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the portable firmware logic check")
    source = Path(__file__).with_name("fall_detector_check.cpp")
    sketch = source.parent.parent / "firmware/thirdeye_ai_device/thirdeye_ai_device.ino"
    _, start, remaining = sketch.read_text(encoding="utf-8").partition("// BEGIN PORTABLE FALL DETECTOR")
    detector, end, _ = remaining.partition("// END PORTABLE FALL DETECTOR")
    assert start and end, "The test must compile the detector from the actual sketch"
    translation_unit = "#include <cmath>\n#include <cstdint>\n" + detector + source.read_text(encoding="utf-8")
    executable = tmp_path / "fall_detector_check.exe"
    subprocess.run([compiler, "-std=c++11", "-Wall", "-Wextra", "-pedantic", "-static",
                    "-x", "c++", "-", "-o", str(executable)], input=translation_unit,
                   check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, timeout=10)


def test_fall_websocket_validates_acknowledges_and_deduplicates():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    camera = Esp32Camera(host="127.0.0.1", port=port, token="test-token")
    event = {"type": "fall_suspected", "event_id": "abcdef01-1", "uptime_ms": 5000,
             "peak_g": 3.0, "peak_gyro": 300.0, "tilt_deg": 85.0}
    status = {"type": "imu_status", "uptime_ms": 10000, "samples": 1000,
              "read_failures": 0, "queue_drops": 0, "sensor_ok": True}
    try:
        with connect(f"ws://127.0.0.1:{port}/ws/imu?token=test-token") as device:
            for invalid in (None, [], {**event, "peak_g": float("nan")},
                            {**event, "tilt_deg": 200}, {**event, "event_id": "invalid"},
                            {**event, "peak_gyro": True}, {**status, "sensor_ok": "yes"}):
                device.send(json.dumps(invalid))
            for _ in range(2):
                device.send(json.dumps(event))
                assert device.recv(timeout=2) == "ACK:abcdef01-1"
            assert camera.imu_monitor["fall_count"] == 1
            device.send(json.dumps(status))
            deadline = time.monotonic() + 2
            while camera.imu_monitor["status"] is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert camera.imu_monitor["status"]["samples"] == 1000
            assert camera.imu_monitor["age"] < 2
            snapshot = camera.imu_monitor
            snapshot["latest_fall"]["peak_g"] = 99
            assert camera.imu_monitor["latest_fall"]["peak_g"] == 3.0
        # Reconnection retries use the same event ID and must still be acknowledged.
        deadline = time.monotonic() + 2
        while camera.imu_monitor["connected"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not camera.imu_monitor["connected"]
        with connect(f"ws://127.0.0.1:{port}/ws/imu?token=test-token") as device:
            device.send(json.dumps(event))
            assert device.recv(timeout=2) == "ACK:abcdef01-1"
            assert camera.imu_monitor["fall_count"] == 1
    finally:
        camera.release()
