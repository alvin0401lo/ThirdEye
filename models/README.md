# Model files

Model weights are downloaded during setup or first use and are not committed to the repository.

- `deployment/ubuntu/ubuntu_setup.sh` downloads MediaPipe's `hand_landmarker.task` into this directory.
- Ultralytics downloads the selected YOLOE checkpoint and prompt encoder on first use.
- Local Whisper and other optional models may download weights on first use.

Initial setup needs internet access and disk space. Review model and dependency licenses before redistribution.
