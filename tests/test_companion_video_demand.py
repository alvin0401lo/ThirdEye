import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

import pytest
from websockets.sync.client import connect

from thirdeye.network_camera import Esp32Camera


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_website_requests_video_only_after_user_joins():
    website = Path(__file__).resolve().parents[2] / "nexus"
    node = shutil.which("node")
    if node is None or not (website / "server.js").exists():
        pytest.skip("Companion website and Node.js are required")
    bridge_port, camera_port = free_port(), free_port()
    environment = {**os.environ, "PORT": str(free_port()), "LOCAL_PORT": str(free_port()),
                   "BRIDGE_PORT": str(bridge_port)}
    server = subprocess.Popen([node, "server.js"], cwd=website, env=environment,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    camera = None
    try:
        deadline = time.monotonic() + 8
        while True:
            try:
                with socket.create_connection(("127.0.0.1", bridge_port), timeout=0.2):
                    break
            except OSError:
                if server.poll() is not None or time.monotonic() >= deadline:
                    raise AssertionError("Companion test server failed to start")
                time.sleep(0.05)
        camera = Esp32Camera(host="127.0.0.1", port=camera_port,
                             companion_bridge_url=f"ws://127.0.0.1:{bridge_port}")
        with connect(f"ws://127.0.0.1:{camera_port}/ws/camera") as device:
            assert device.recv(timeout=2) == "VIDEO:OFF"
            with connect(f"ws://127.0.0.1:{bridge_port}") as viewer:
                with pytest.raises(TimeoutError):
                    device.recv(timeout=0.5)
                viewer.send(json.dumps({"type": "join", "room": "THIRDEYE"}))
                assert device.recv(timeout=4) == "VIDEO:PREVIEW"
            assert device.recv(timeout=3) == "VIDEO:OFF"
            with camera.video_session():
                assert device.recv(timeout=2) == "VIDEO:ON"
                with connect(f"ws://127.0.0.1:{bridge_port}") as viewer:
                    viewer.send(json.dumps({"type": "join", "room": "THIRDEYE"}))
                    assert json.loads(viewer.recv(timeout=2))["type"] == "peers"
                with pytest.raises(TimeoutError):
                    device.recv(timeout=0.5)
                with camera.video_session(profile="fast"):
                    assert device.recv(timeout=2) == "VIDEO:FAST"
                assert device.recv(timeout=2) == "VIDEO:ON"
            assert device.recv(timeout=2) == "VIDEO:OFF"
        # Fall relay must continue even without a connected camera socket.
        def wait_monitor(viewer, predicate):
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline:
                message = json.loads(viewer.recv(timeout=4))
                if message.get("type") == "fall_monitor" and predicate(message):
                    return message
            raise AssertionError("Expected fall monitor update did not arrive")

        with connect(f"ws://127.0.0.1:{camera_port}/ws/imu") as imu:
            imu.send(json.dumps({"type": "imu_status", "uptime_ms": 10000, "samples": 1000,
                                 "read_failures": 0, "queue_drops": 0, "sensor_ok": True}))
            event = {"type": "fall_suspected", "event_id": "deadbeef-1", "uptime_ms": 11000,
                     "peak_g": 3.0, "peak_gyro": 250.0, "tilt_deg": 70.0}
            imu.send(json.dumps(event))
            assert imu.recv(timeout=2) == "ACK:deadbeef-1"
            imu.send(json.dumps(event))
            assert imu.recv(timeout=2) == "ACK:deadbeef-1"
            with connect(f"ws://127.0.0.1:{bridge_port}") as viewer:
                viewer.send(json.dumps({"type": "join", "room": "THIRDEYE"}))
                monitor = wait_monitor(viewer, lambda msg: msg["fall_count"] == 1)
                assert monitor["connected"] and monitor["sensor_ok"] and not monitor["stale"]
                assert monitor["latest_fall"]["peak_g"] == 3.0
                assert monitor["latest_fall"]["received_at"] > 0
                # A viewer cannot impersonate the trusted local device relay.
                viewer.send(json.dumps({**monitor, "fall_count": 999}))
                imu.close()
                offline = wait_monitor(viewer, lambda msg: not msg["connected"])
                assert offline["fall_count"] == 1
                assert offline["latest_fall"]["event_id"] == "deadbeef-1"
    finally:
        if camera is not None:
            camera.release()
        server.terminate()
        try:
            server.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.communicate(timeout=5)
