# Deployment

Choose the instructions for the operating system that will run the ThirdEye AI service. Run commands from the repository root.

## Requirements

- Python 3.11–3.13
- The AI host and ESP32 connected to the same local network
- A valid OpenAI API key and model IDs
- A private device token shared between `.env` and the ESP32 firmware

## Windows

Open PowerShell in the repository root and install the project:

```powershell
.\deployment\windows\windows_setup.ps1
```

If PowerShell blocks the script, run it for this session:

```powershell
powershell -ExecutionPolicy Bypass -File .\deployment\windows\windows_setup.ps1
```

Edit `.env` in the repository root. Set `OPENAI_API_KEY`, the model IDs, and `THIRDEYE_CAMERA_TOKEN`.

Run each device check separately, then start voice control:

```powershell
.\deployment\windows\windows_run.ps1 --device-test
.\deployment\windows\windows_run.ps1 --mic-test
.\deployment\windows\windows_run.ps1 --speaker-test
.\deployment\windows\windows_run.ps1 --voice-control
```

## Ubuntu

Open a terminal in the repository root and install the system and Python dependencies:

```bash
chmod +x deployment/ubuntu/*.sh
./deployment/ubuntu/ubuntu_setup.sh
```

Edit `.env` in the repository root. Set `OPENAI_API_KEY`, the model IDs, and `THIRDEYE_CAMERA_TOKEN`.

Run each device check separately, then start voice control:

```bash
./deployment/ubuntu/ubuntu_run.sh --device-test
./deployment/ubuntu/ubuntu_run.sh --mic-test
./deployment/ubuntu/ubuntu_run.sh --speaker-test
./deployment/ubuntu/ubuntu_run.sh --voice-control
```

On Ubuntu with systemd, starting voice control installs and starts the ThirdEye service, which is configured to start after reboot.

## ESP32 setup

In `firmware/thirdeye_ai_device/thirdeye_ai_device.ino`, set the Wi-Fi name and password, the AI host's local IP address, and a private `DEVICE_TOKEN`. The token must match `THIRDEYE_CAMERA_TOKEN` in `.env`. Compile and flash the firmware after changing these values.

The ESP32 and AI host must be on the same local network. Allow inbound TCP port `8081` on the AI host if its firewall is enabled. Do not expose this device service directly to the public internet.

## Website and Telegram alerts

The companion website and Telegram alert service are separate from the AI deployment. Follow the setup instructions in the repository's `companion/` directory.
