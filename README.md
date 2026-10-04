# ThirdEye

ThirdEye is a voice-guided visual assistance prototype. An ESP32-S3 camera captures the wearer's view while the AI service handles voice requests, object finding, scene descriptions, and spoken guidance. A companion website lets invited viewers open the live camera feed from an alert link.

The proposal frames ThirdEye as an effort to make assistive technology more accessible to people with visual impairment. This repository covers the current glasses prototype; the proposed haptic stick, ToF obstacle mapping, and GPS features remain future work.

> ThirdEye is a prototype. Object guidance and suspected-fall detection are not safety-certified and must not replace human assistance.

## Features

- Wake-word interaction with “Hi, Third Eye”.
- Voice requests for finding objects, describing a scene, and asking questions about the camera view.
- Object segmentation and hand tracking for object-finding guidance.
- Spoken directional cues through the speaker.
- Follow-up listening after visual responses; saying “done” returns to wake-word mode.
- On-demand video streaming, with camera upload paused when no visual task or joined viewer needs it.
- Suspected-fall alerts sent to Telegram with a link to the companion website. The room is prefilled; the recipient must click **Join room** to view the camera.

## Hardware

| Part | Configuration |
|---|---|
| Glasses controller | ESP32-S3-CAM N16R8 |
| Camera | OV2640 |
| Microphone | INMP441 I2S |
| Speaker amplifier | MAX98357A |
| Motion sensor | MPU6500 |
| AI service host | Linux computer on the same local network |

The firmware uses the existing board pin map. Check [firmware/README.md](firmware/README.md) and confirm your board revision and wiring before flashing.

## Software and AI models

- **Software:** Python 3.11–3.13, ESP32 Arduino Core, Markus Sattler's WebSockets library, and Node.js for the companion website.
- **AI service:** FastAPI, OpenCV, Ultralytics, MediaPipe, faster-whisper, and the OpenAI Python SDK.
- **Object finding:** YOLOE segmentation; the example configuration selects `yoloe-26s-seg.pt` at image size 640.
- **Hand tracking:** MediaPipe Hand Landmarker.
- **Speech:** faster-whisper or the configured transcription service; the example configuration uses `gpt-live-transcribe`.
- **Command interpretation and visual questions:** OpenAI models selected with `THIRDEYE_OPENAI_MODEL` and `THIRDEYE_VISION_MODEL`.

The deployment setup downloads the MediaPipe hand model; Ultralytics and other optional model weights download on first use. Model files are not included in the repository. Check model and dependency licenses before redistribution.

## Project structure

```text
audio_cues/                       Spoken navigation cues
companion/                        Website, live viewer, and Telegram alert service
deployment/                       Setup, launch, and service files
firmware/thirdeye_ai_device/      ESP32-S3 integrated firmware
models/                           Model download notes
src/thirdeye/                     AI service and application modules
tests/                            Python and firmware regression checks
.env.example                      AI service configuration template
requirements.txt                  Python dependencies
constraints.txt                   Pinned dependency constraints
pyproject.toml                    Python package metadata
```

## Deployment

The AI service is intended to run on a Linux computer near the glasses so it can access the camera, microphone, speaker, and local network while handling AI workloads. From the cloned project directory, start voice control with:

```bash
./deployment/run.sh --voice-control
```

On first launch, the script prepares the Python environment, creates `.env`, and asks for missing API/model settings and the private device token. It may ask for administrator credentials to install system packages. On a system with systemd, it installs and starts the service so it starts again after reboot. Flash the firmware once with the local Wi-Fi details, AI host address, and the same private token entered in `.env`.

To enable the website and Telegram alerts, configure `companion/.env` from `companion/.env.example`, including `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, and a publicly reachable HTTPS `PUBLIC_URL`. Deploy the companion service at that address with WebSocket support, then install and start it from `companion/`:

```bash
cd companion
npm ci
npm start
```

The alert link opens the website with the room code filled in. Viewers join only after clicking **Join room**. Keep the AI service's device port on a trusted local network; do not expose it directly to the public internet.

## Privacy and limitations

Keep `.env` files, API keys, device tokens, logs, recordings, virtual environments, caches, and downloaded model weights out of GitHub. Visual questions and cloud transcription may send camera images or audio to the configured cloud service; get consent and avoid sensitive scenes in demonstrations. Suspected-fall alerts are heuristic and may be missed or triggered incorrectly. The project uses the MIT License; review model and dependency terms before redistribution.