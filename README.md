# ThirdEye

PC-hosted assistive AI prototype with ESP32-S3 camera, optional device audio
and MPU6500 suspected-fall monitoring. The companion website is a separate
project and is not included in this snapshot.

## Features

- Local wake phrase: "Hi, Third Eye".
- Streaming transcription; partial text is displayed, only final text executes.
- Segmentation, hand tracking and directional audio for object finding.
- Scene descriptions, text-reading questions and visual follow-up questions.
- Follow-up listening with a 30-second speech-start timeout.
- Separate ESP32 camera capture/send tasks and timing diagnostics.
- On-demand video: visual tasks or an explicit website room join enable streaming;
  standby pauses live JPEG upload while audio and fall monitoring continue.
- Local suspected-fall events with receipt acknowledgments and powered retry.
- Companion fall-monitor updates are relayed independently of the camera socket,
  with duplicate-event protection and sensor connection/staleness status.

Object-finding "done" is manual completion, not verified grasp detection.
Suspected-fall events are not confirmed human falls.

## Repository Layout

```text
src/thirdeye/          Application, AI, audio and camera modules
firmware/             Current single-file integrated device sketch
audio_cues/           Short navigation WAV assets
models/README.md      Model download notes; weights are not committed
.env.example          Public configuration template
setup.ps1             Windows setup
run.ps1               Terminal launcher
Start ThirdEye.*      Optional Windows background launcher
```

## Windows Setup

Install Python 3.11-3.13 and Git, then run:

```powershell
.\setup.ps1
Copy-Item .env.example .env
```

Fill `.env` with your API key, routing/vision model IDs supported by your
account, and a private device token matching the firmware. Never commit it.
Internet/API billing are required for cloud tasks, not camera/IMU diagnostics.
Initial model downloads require internet and disk space.

The default architecture uses ESP32 video/IMU and PC microphone/speaker.
For a USB camera, set `THIRDEYE_CAMERA_SOURCE=webcam` and its camera index.
Width/height environment settings apply to USB cameras, not ESP32 resolution.

## Firmware and Testing

Read [firmware/README.md](firmware/README.md) for the main sketch configuration.
The current single-file firmware is
`firmware/thirdeye_ai_device/thirdeye_ai_device.ino`.
It targets the existing ESP32-S3-CAM N16R8 / OV2640 wiring, not all S3 boards.
Verify physical pin availability. Install the Espressif ESP32 Arduino core
and WebSockets by Markus Sattler. For verified N16R8 hardware, select
ESP32S3 Dev Module, 16MB Flash and OPI PSRAM.

Configure sketch Wi-Fi credentials, PC LAN IPv4 address, port 8081 and token.
`PC_AUDIO_CAMERA_TEST=1` disables device audio for PC-audio tests; `0` enables it.
MPU6500 uses GPIO21 SDA, GPIO47 SCL, 3.3V, common GND and AD0 GND (address 0x68).
Audio clocks use GPIO2 BCLK and GPIO1 WS; mic SD uses GPIO3, speaker DIN GPIO14.

Run one server at a time; PC and ESP32 must be mutually reachable:

```powershell
.\run.ps1 --camera-test
.\run.ps1 --imu-test
.\run.ps1 --device-test
.\run.ps1 --voice-control
```

For ESP32 microphone/speaker tests, flash with device audio enabled:

Use the matching `HTTP_WAV_V1` integrated firmware. Microphone upload remains
WebSocket; speaker audio uses an independent authenticated HTTP WAV download
on port 8081. `--speaker-test` does not require microphone packets and plays
only `camera_down.wav`, then checks the board completion report. Listen to
confirm audible sound. Older WebSocket speaker firmware is not compatible.

```powershell
$env:THIRDEYE_AUDIO_SOURCE = 'esp32'
.\run.ps1 --mic-test
.\run.ps1 --speaker-test
```

Workflow: local wake -> final command transcription -> command routing and
visual analysis -> spoken answer -> follow-up listening. Standalone "done"
returns to standby; object finding has a dedicated local completion listener.
This runtime-only snapshot excludes automated tests, standalone hardware
sketches and old development variants. They remain in the original development
project. Camera stutter remains under investigation; record `camera_perf` before
claiming a performance improvement. Physical firmware/audio/fall validation is
still required.

## Security and Limitations

Publish this prepared snapshot, not the original development directory.
Credentials, recordings, logs, environments, caches and weights are excluded;
exported firmware credentials are placeholders. Device WebSockets are token
protected but unencrypted: use a trusted LAN, not a public port forward.
Cloud vision/transcription can upload personal images/audio; obtain consent
and use non-sensitive scenes for public demos.

Fall detection is an unvalidated glasses-mounted heuristic. Dropped glasses
can cause false events and actual falls can be missed. Event queues and PC
duplicate history are RAM-only; receipt ACK is not wearer confirmation.
There is no wearer sensor, user-safety confirmation or automatic contact alert.
Do not test by making a person fall or rely on this as a safety device.

No project license has been selected. Review dependency/model licenses and
choose a project license before presenting this as reusable open-source work.
