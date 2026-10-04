from thirdeye.network_camera import decode_microphone_packet, is_valid_audio_pcm, is_valid_camera_jpeg
from thirdeye.network_camera import Esp32Camera
from collections import deque
import threading
import socket
import time
import numpy as np
import pytest
import cv2
from websockets.sync.client import connect
from websockets.sync.server import serve

from thirdeye.camera import CameraError


def test_camera_jpeg_validation() -> None:
    assert is_valid_camera_jpeg(b"\xff\xd8frame\xff\xd9")
    assert not is_valid_camera_jpeg(b"not a jpeg")
    assert not is_valid_camera_jpeg(b"\xff\xd8truncated")


def test_camera_decode_rotates_pixels_counterclockwise():
    source = np.zeros((48, 64, 3), dtype=np.uint8)
    source[:24, :32] = (30, 120, 240)
    _, jpeg = cv2.imencode(".jpg", source)
    decoded = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    assert np.array_equal(Esp32Camera._decode(jpeg.tobytes()), cv2.rotate(decoded, cv2.ROTATE_90_COUNTERCLOCKWISE))


def test_video_demand_survives_reconnect_and_task_failure():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    camera = Esp32Camera(host="127.0.0.1", port=port, timeout=2)
    try:
        with connect(f"ws://127.0.0.1:{port}/ws/camera") as device:
            assert device.recv(timeout=2) == "VIDEO:OFF"
            with pytest.raises(RuntimeError, match="failed task"):
                with camera.video_session():
                    assert device.recv(timeout=2) == "VIDEO:ON"
                    with camera.video_session():
                        camera._set_companion_viewing(True)
                    raise RuntimeError("failed task")
            assert camera._video_users == 0
            assert camera._video_enabled  # Website still needs video.
        deadline = time.monotonic() + 2
        while camera._connected and time.monotonic() < deadline:
            time.sleep(0.01)
        with connect(f"ws://127.0.0.1:{port}/ws/camera") as device:
            assert device.recv(timeout=2) == "VIDEO:PREVIEW"
            camera._set_companion_viewing(False)
            assert device.recv(timeout=2) == "VIDEO:OFF"
            assert camera._video_users == 0
    finally:
        camera.release()


def test_shared_video_demand_uses_highest_profile_and_restores_previous_user():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    camera = Esp32Camera(host="127.0.0.1", port=port, timeout=2)
    try:
        with connect(f"ws://127.0.0.1:{port}/ws/camera") as device:
            assert device.recv(timeout=2) == "VIDEO:OFF"
            camera._set_companion_viewing(True)
            assert device.recv(timeout=2) == "VIDEO:PREVIEW"
            with camera.video_session(profile="live"):
                assert device.recv(timeout=2) == "VIDEO:ON"
                with camera.video_session(profile="fast"):
                    assert device.recv(timeout=2) == "VIDEO:FAST"
                assert device.recv(timeout=2) == "VIDEO:ON"
            assert device.recv(timeout=2) == "VIDEO:PREVIEW"
            camera._set_companion_viewing(False)
            assert device.recv(timeout=2) == "VIDEO:OFF"
    finally:
        camera.release()


def test_microphone_pcm_validation() -> None:
    assert is_valid_audio_pcm(b"\x00\x00")
    assert is_valid_audio_pcm(b"\x00\x00" * 160)
    assert not is_valid_audio_pcm(b"")
    assert not is_valid_audio_pcm(b"\x00")


def _microphone_packet(sequence: int, value: int = 1) -> bytes:
    return b"TEA1" + sequence.to_bytes(4, "little") + value.to_bytes(2, "little") * 320


def test_microphone_packet_decoder_preserves_legacy_pcm() -> None:
    assert decode_microphone_packet(_microphone_packet(7)) == (7, b"\x01\x00" * 320)
    assert decode_microphone_packet(b"\x01\x00" * 320) == (None, b"\x01\x00" * 320)
    assert decode_microphone_packet(b"TEA1\x00") is None


def test_8k_microphone_packet_is_upsampled_for_16k_speech_pipeline() -> None:
    pcm = np.array([0, 1000, -1000, 0] * 40, dtype="<i2").tobytes()
    packet = b"TEA2" + (7).to_bytes(4, "little") + pcm
    sequence, converted = decode_microphone_packet(packet)

    assert sequence == 7
    assert len(converted) == 640
    assert np.frombuffer(converted, dtype="<i2")[:8].tolist() == [0, 500, 1000, 0, -1000, -500, 0, 0]
    assert decode_microphone_packet(b"TEA2\x00") is None


def test_audio_websocket_accepts_8k_microphone_packet() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    camera = Esp32Camera(host="127.0.0.1", port=port, timeout=2)
    try:
        with connect(f"ws://127.0.0.1:{port}/ws/audio") as device:
            device.send(b"TEA2" + (1).to_bytes(4, "little") + b"\x01\x00" * 160)
            assert camera.read_audio(timeout=2) == b"\x01\x00" * 320
            assert camera.audio_integrity == ("sequenced-8k", 0, 0, 0)
    finally:
        camera.release()


def test_audio_websocket_detects_missing_sequence_and_receiver_overflow() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    camera = Esp32Camera(host="127.0.0.1", port=port, timeout=2)
    camera._audio_chunks = deque(maxlen=2)
    try:
        with connect(f"ws://127.0.0.1:{port}/ws/audio") as device:
            for sequence in (10, 11, 13):
                device.send(_microphone_packet(sequence, sequence))
            deadline = time.monotonic() + 2
            while camera.audio_integrity[3] == 0 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert camera.audio_integrity == ("sequenced", 2, 1, 1)
            assert camera.read_audio(timeout=0) == (11).to_bytes(2, "little") * 320
            assert camera.read_audio(timeout=0) == (13).to_bytes(2, "little") * 320
    finally:
        camera.release()


def test_stalled_audio_reconnect_does_not_let_old_cleanup_clear_new_connection() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    camera = Esp32Camera(host="127.0.0.1", port=port, timeout=2)
    try:
        with connect(f"ws://127.0.0.1:{port}/ws/audio") as old:
            deadline = time.monotonic() + 2
            while not camera.audio_connected and time.monotonic() < deadline:
                time.sleep(0.01)
            old.send("DEVICE_AUDIO:HTTP_WAV_V1")
            deadline = time.monotonic() + 2
            while camera._audio_firmware is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert camera._audio_firmware == "HTTP_WAV_V1"
            camera._audio_last_pcm_at = time.monotonic() - 6
            with connect(f"ws://127.0.0.1:{port}/ws/audio") as replacement:
                replacement.send("DEVICE_AUDIO:ACK_V2")
                replacement.send(_microphone_packet(1))
                assert camera.read_audio(timeout=2) == b"\x01\x00" * 320
                time.sleep(0.1)
                assert camera.audio_connected
                with pytest.raises(Exception):
                    old.recv(timeout=1)
                replacement.send(_microphone_packet(2))
                assert camera.read_audio(timeout=2) == b"\x01\x00" * 320
                with pytest.raises(Exception):
                    with connect(f"ws://127.0.0.1:{port}/ws/audio"):
                        pass
    finally:
        camera.release()


def test_audio_disconnect_logs_close_code_reason_and_duration(capsys) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    camera = Esp32Camera(host="127.0.0.1", port=port, timeout=2)
    try:
        with connect(f"ws://127.0.0.1:{port}/ws/audio") as device:
            device.send(_microphone_packet(1))
            camera.read_audio(timeout=2)
            device.close(code=1001, reason="diagnostic disconnect")
        deadline = time.monotonic() + 2
        while camera.audio_connected and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not camera.audio_connected
    finally:
        camera.release()
    output = capsys.readouterr().out
    assert "microphone disconnected: code=1001" in output
    assert "diagnostic disconnect" in output and "connected_for=" in output


def test_slow_camera_decode_does_not_block_microphone_receive(monkeypatch) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    camera = Esp32Camera(host="127.0.0.1", port=port, timeout=2)
    decoding = threading.Event()
    release_decode = threading.Event()
    def decode(_jpeg):
        decoding.set()
        release_decode.wait(timeout=5)
        return np.zeros((2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(camera, "_decode", decode)
    try:
        with connect(f"ws://127.0.0.1:{port}/ws/camera") as video:
            video.send(b"\xff\xd8frame\xff\xd9")
            assert decoding.wait(timeout=2)
            with connect(f"ws://127.0.0.1:{port}/ws/audio") as audio:
                audio.send(_microphone_packet(1))
                assert camera.read_audio(timeout=1) == b"\x01\x00" * 320
    finally:
        release_decode.set()
        camera.release()


def test_esp32_audio_reader_preserves_packet_order() -> None:
    camera = object.__new__(Esp32Camera)
    camera._audio_condition = threading.Condition()
    camera._audio_chunks = deque([b"\x01\x00", b"\x02\x00"])

    assert camera.read_audio(timeout=0) == b"\x01\x00"
    assert camera.read_audio(timeout=0) == b"\x02\x00"


def test_still_request_matches_response_and_keeps_stream_frame() -> None:
    camera = object.__new__(Esp32Camera)
    camera._condition = threading.Condition()
    camera._still_lock = threading.Lock()
    camera._still_id = 0
    camera._still_result_id = 0
    camera._still_error = False
    camera._still_frame = None
    camera._connected = True
    camera._latest_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    class Loop:
        def call_soon_threadsafe(self, callback, command):
            callback(command)

    class Commands:
        def put_nowait(self, command):
            assert command == "SNAP:1"
            with camera._condition:
                camera._still_frame = np.ones((1200, 1600, 3), dtype=np.uint8)
                camera._still_result_id = 1
                camera._condition.notify_all()

    camera._camera_loop = Loop()
    camera._camera_commands = Commands()

    still = camera.capture_still(timeout=0.1)

    assert still.shape == (1200, 1600, 3)
    assert camera._latest_frame.shape == (480, 640, 3)


def test_still_request_requires_connected_camera() -> None:
    camera = object.__new__(Esp32Camera)
    camera._condition = threading.Condition()
    camera._still_lock = threading.Lock()
    camera._connected = False
    camera._camera_loop = None
    camera._camera_commands = None

    with pytest.raises(CameraError, match="not connected"):
        camera.capture_still()


def test_camera_websocket_still_round_trip() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    camera = Esp32Camera(host="127.0.0.1", port=port, timeout=2)
    try:
        with connect(f"ws://127.0.0.1:{port}/ws/camera") as device:
            assert device.recv(timeout=2) == "VIDEO:OFF"
            _, normal = cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))
            device.send(normal.tobytes())
            assert camera.read().shape == (640, 480, 3)

            result = []
            worker = threading.Thread(target=lambda: result.append(camera.capture_still(timeout=3)))
            worker.start()
            command = device.recv(timeout=2)
            assert command == "SNAP:1"
            _, still = cv2.imencode(".jpg", np.zeros((1200, 1600, 3), dtype=np.uint8))
            device.send("SNAP:1")
            device.send(still.tobytes())
            worker.join(timeout=3)
            assert not worker.is_alive()
            assert result[0].shape == (1600, 1200, 3)
            assert camera._latest_frame.shape == (640, 480, 3)
    finally:
        camera.release()


def test_esp32_camera_relay_forwards_jpeg_to_companion() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        bridge_port = probe.getsockname()[1]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        camera_port = probe.getsockname()[1]
    messages = []
    received = threading.Event()

    def bridge_handler(websocket):
        messages.append(websocket.recv(timeout=3))
        while True:
            message = websocket.recv(timeout=3)
            messages.append(message)
            if isinstance(message, bytes):
                break
        received.set()

    with serve(bridge_handler, "127.0.0.1", bridge_port) as bridge:
        bridge_thread = threading.Thread(target=bridge.serve_forever, daemon=True)
        bridge_thread.start()
        camera = Esp32Camera(
            host="127.0.0.1", port=camera_port, timeout=2,
            companion_bridge_url=f"ws://127.0.0.1:{bridge_port}", companion_room="THIRDEYE",
        )
        try:
            with connect(f"ws://127.0.0.1:{camera_port}/ws/camera") as device:
                _, frame = cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))
                device.send(frame.tobytes())
                assert received.wait(4)
                assert '"type": "join_camera_relay"' in messages[0]
                assert messages[-1] == frame.tobytes()
        finally:
            camera.release()
            bridge.shutdown()
            bridge_thread.join(timeout=3)
