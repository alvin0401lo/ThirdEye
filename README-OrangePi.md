# ThirdEye on Orange Pi 5 Pro

This is an Orange Pi deployment copy of the latest working source from `ai/_model_pipeline`. The Orange Pi runs the AI server; the ESP32 remains the camera, INMP441 microphone, MAX98357A speaker, and MPU6500 event source. Audio transcription and visual answers use OpenAI, so those features need internet access and a valid API key.

## Install

Use the Ubuntu 24.04 desktop image, connect the Orange Pi and ESP32 to the same LAN, and open a terminal in this folder:

```bash
chmod +x setup_orangepi.sh run_orangepi.sh
./setup_orangepi.sh
nano .env
```

In `.env`, set the real `OPENAI_API_KEY` and model IDs supported by your API account. The Orange Pi profile is already configured for ESP32 camera/audio, `gpt-live-transcribe`, YOLOE nano segmentation, and the English `base.en` wake model.

Edit `firmware/thirdeye_ai_device.ino`: set `SERVER_HOST` to the Orange Pi's LAN IPv4 address and set a private `DEVICE_TOKEN`. Put the same token in `THIRDEYE_CAMERA_TOKEN` in `.env`, then compile and flash that firmware to the ESP32 if the board is not already running the matching `HTTP_WAV_V1` build.

Start with hardware checks, one at a time:

```bash
./run_orangepi.sh --device-test
./run_orangepi.sh --mic-test
./run_orangepi.sh --speaker-test
./run_orangepi.sh --voice-control
```

Allow inbound TCP port `8081` on the Orange Pi if its firewall is enabled. Say `Hi, Third Eye`, wait for the ready cue, then try `Find my bottle`, `Describe scene`, or `What is in front of me?`. Partial transcription is display-only; a final transcript is required before a command runs. After a visual answer, follow-up listening lasts 30 seconds; standalone `done` returns to wake-word mode.

## Model and platform limits

- The starting profile uses `yoloe-26n-seg.pt` at 512 input size and `base.en` to keep CPU load lower. Set the model path back to `yoloe-26s-seg.pt` and input size to 640 if accuracy matters more than latency.
- The current YOLOE/PyTorch path runs on the CPU. It does **not** use the RK3588 NPU. RKNN acceleration is a separate model-export/inference project; do not expect a speed-up just because this is an Orange Pi 5 Pro.
- MediaPipe 1.0.1 provides an ARM64 wheel and the HandLandmarker Tasks API used by this code, but PyPI classifies this release as Alpha. The wheel can be installed; hand tracking still needs an on-board smoke test.
- The installer installs MediaPipe without its declared OpenCV-contrib package because this app only uses the Tasks API and standard `cv2`; keeping one OpenCV package avoids clashing `cv2` files.
- Speech synthesis uses `espeak-ng` on Linux. With `THIRDEYE_AUDIO_SOURCE=esp32`, speech and WAV cues are sent to the ESP32 speaker. PC audio can be selected in `.env` for desk testing; it requires a working ALSA/PortAudio device.
- First installation downloads large Python/model packages and model checkpoints. Use a reliable power supply and preferably an SSD; installation to microSD is possible but slower and writes more data.
- The website is not bundled here. This folder is the AI/device server only; deploy the separate web project if you need its browser video-call page.

## Known acceptance boundary

Host tests verify routing, audio conversion, and server logic. They do not prove Orange Pi package resolution, NPU performance, physical camera quality, ESP32 Wi-Fi stability, or microphone/speaker behavior. The first board run must verify imports and `--device-test`, then test microphone, speaker, wake phrase, and one visual request separately.
