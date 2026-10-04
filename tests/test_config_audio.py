import pytest

from thirdeye.config import Settings


@pytest.mark.parametrize("source", ["esp32", "auto"])
def test_board_audio_does_not_resolve_pc_device_names(monkeypatch, source):
    monkeypatch.setattr("thirdeye.config.load_dotenv", lambda path: None)
    monkeypatch.setenv("THIRDEYE_CAMERA_SOURCE", "esp32")
    monkeypatch.setenv("THIRDEYE_AUDIO_SOURCE", source)
    monkeypatch.setenv("THIRDEYE_MICROPHONE_DEVICE", "esp32")
    monkeypatch.setenv("THIRDEYE_SPEAKER_DEVICE", "esp32")
    settings = Settings.load()
    assert settings.microphone_device is None
    assert settings.speaker_device is None


def test_pc_audio_rejects_board_name_with_configuration_guidance(monkeypatch):
    monkeypatch.setattr("thirdeye.config.load_dotenv", lambda path: None)
    monkeypatch.setenv("THIRDEYE_AUDIO_SOURCE", "pc")
    monkeypatch.setenv("THIRDEYE_MICROPHONE_DEVICE", "esp32")
    with pytest.raises(ValueError, match="Set THIRDEYE_AUDIO_SOURCE=esp32"):
        Settings.load()


def test_pc_audio_still_accepts_real_device_selectors(monkeypatch):
    monkeypatch.setattr("thirdeye.config.load_dotenv", lambda path: None)
    monkeypatch.setenv("THIRDEYE_AUDIO_SOURCE", "pc")
    monkeypatch.setenv("THIRDEYE_MICROPHONE_DEVICE", "2")
    monkeypatch.setenv("THIRDEYE_SPEAKER_DEVICE", "USB Audio")
    settings = Settings.load()
    assert settings.microphone_device == 2
    assert settings.speaker_device == "USB Audio"
