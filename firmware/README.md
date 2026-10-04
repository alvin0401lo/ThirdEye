# Integrated Device Firmware

Open `thirdeye_ai_device/thirdeye_ai_device.ino` in Arduino IDE. Camera,
microphone, speaker, MPU6500, suspected-fall logic and WebSocket transport are
all in this one sketch. No extra local header is required.

Configure Wi-Fi credentials, PC IPv4 address, port 8081 and a private token
matching `THIRDEYE_CAMERA_TOKEN` in the PC `.env` file. Exported values are
placeholders. Do not commit real credentials.

Install the Espressif ESP32 Arduino core and WebSockets by Markus Sattler.
The sketch targets the existing ESP32-S3-CAM N16R8 / OV2640 pin map. Verify
your physical board pins before wiring. For verified N16R8 hardware, use
ESP32S3 Dev Module, 16MB Flash and OPI PSRAM.

- MPU6500: SDA GPIO21, SCL GPIO47, 3.3V, common GND, AD0 GND.
- INMP441: SCK GPIO2, WS GPIO1, SD GPIO3, VDD 3.3V, GND and L/R GND.
- MAX98357A: BCLK GPIO2, LRC GPIO1, DIN GPIO14, VIN 5V, common GND.
- `PC_AUDIO_CAMERA_TEST=1` disables ESP32 audio while retaining camera/IMU.
- `ENABLE_IMU=1` enables local suspected-fall monitoring.

Audio firmware version: `HTTP_WAV_V1`. Microphone upload uses WebSocket;
speaker output independently polls `GET /stream.wav` on the same port 8081.
The PC serves one finite mono PCM16 16 kHz WAV per utterance, with a playback
ID. The board downloads the entire WAV into PSRAM before I2S playback,
then posts `done` or `error` to `/speaker/status`. Both HTTP routes require
the configured token. This removes the old WebSocket START/PCM/STOP path.
A reported completion proves firmware writes, not audible sound.
Download inactivity is limited to 3 seconds, total download to 15 seconds,
and each I2S write to 100 ms. Audio is limited to 60 seconds per utterance;
without PSRAM, longer clips may fail allocation. Download-before-play adds
startup latency but avoids network-caused playback underruns.
Microphone upload remains paused during actual playback; not full duplex.

This follows OpenAIglasses_for_Navigation's WebSocket microphone / independent
HTTP WAV speaker split. It does not copy its PDM wiring or persistent chunked
stream: this board retains INMP441 and its existing shared 16 kHz I2S clocks.
Use a trusted LAN; this local HTTP transport is not encrypted.

Camera initialization uses VGA with PSRAM, JPEG quality 10 and an 88 ms target
interval. Without PSRAM it uses QVGA, JPEG quality 12. Text-photo requests
temporarily switch to UXGA. The serial `camera_perf` line reports capture,
copy, send and queue timing; actual smoothness has not been guaranteed.

Live video defaults to paused. The matching PC application sends `VIDEO:ON`
for object finding, visual capture, camera/device tests, or a website viewer
who explicitly joins the room. It sends `VIDEO:OFF` when no task or viewer
needs video. Camera WebSocket remains connected for control and UXGA `SNAP`
requests; microphone and fall monitoring continue while video is paused.
Reconnections synchronize the current demand. Flash this sketch and update
the PC application together; old firmware does not support video control.
Speaker idle HTTP polling uses a 250 ms delay.

Standalone tests and legacy firmware are intentionally excluded from this
runtime-only snapshot and remain in the original local development project.
Suspected-fall monitoring is not certified human-fall detection.
