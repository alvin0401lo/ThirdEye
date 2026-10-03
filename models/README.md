# Downloaded Models

Do not commit model weights or TorchScript encoders.

- `setup.ps1` downloads `hand_landmarker.task` from the MediaPipe model bucket.
- Ultralytics downloads the selected YOLOE checkpoint and prompt encoder on first use.
- Local Whisper and ZoeDepth may download weights on their first run.

Initial setup needs internet, disk space and compatible dependencies. Review
upstream licenses before redistributing model files.
