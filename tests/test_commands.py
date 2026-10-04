from types import SimpleNamespace

from thirdeye.ai_service import AIService, _local_command, is_english_command_text
from thirdeye.models import Intent


def test_find_my_bottle():
    command = _local_command("Help me find my water bottle")
    assert command.intent is Intent.FIND_OBJECT
    assert command.target == "water bottle"


def test_incomplete_find_command_has_no_target():
    service = AIService(None, "command-model", "transcribe-model")
    for text in ("Find a...", "Find my", "Find the..."):
        command = service.understand_command(text)
        assert command.intent is Intent.FIND_OBJECT
        assert command.target is None


def test_transcription_command_list_is_not_executed():
    service = AIService(None, "command-model", "transcribe-model")
    command = service.understand_command(
        "Hi Third Eye. Find my bottle. Find my keys. Find my phone. "
        "Describe the scene. What do you see? Shutdown Third Eye."
    )
    assert command.intent is Intent.UNKNOWN


def test_transcription_does_not_prime_model_with_commands(tmp_path):
    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"audio")
    received = {}

    def create(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(text="Find my bottle")

    service = AIService(None, "command-model", "transcribe-model")
    service._client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    assert service.transcribe(str(audio_path)) == "Find my bottle"
    assert "prompt" not in received


def test_chinese_target_is_normalized_for_yoloe():
    command = _local_command("帮我找一下水瓶")
    assert command.intent is Intent.FIND_OBJECT
    assert command.target == "water bottle"


def test_describe_scene():
    assert _local_command("Describe scene").intent is Intent.DESCRIBE_SCENE


def test_visual_questions_do_not_become_generic_scene_descriptions():
    for question in (
        "What do you see?",
        "What is in front of me?",
        "What does the sign say?",
        "What am I holding?",
        "What kind of area is this?",
        "What is on my left?",
        "What color is it?",
        "What is ahead of me?",
        "What does this label say?",
        "Which section am I in?",
        "Is there a door ahead?",
    ):
        assert _local_command(question).intent is Intent.VISUAL_QUESTION


def test_unrelated_question_is_not_routed_to_camera():
    assert _local_command("What time is it?").intent is Intent.UNKNOWN
    assert _local_command("Is there a way to reset my password?").intent is Intent.UNKNOWN


def test_common_commands_skip_cloud_intent_classification():
    service = AIService(None, "command-model", "transcribe-model")
    service._client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: None))

    assert service.understand_command("Find my bottle").intent is Intent.FIND_OBJECT
    assert service.understand_command("Describe the scene").intent is Intent.DESCRIBE_SCENE
    assert service.understand_command("What does this label say?").intent is Intent.VISUAL_QUESTION


def test_exit_returns_to_standby():
    assert _local_command("exit").intent is Intent.STANDBY


def test_explicit_shutdown_exits_voice_control():
    assert _local_command("shutdown third eye").intent is Intent.EXIT


def test_english_voice_command_is_accepted():
    assert is_english_command_text("Find my water bottle.")


def test_chinese_voice_command_is_rejected():
    assert not is_english_command_text("\u5e2e\u6211\u627e\u6c34\u74f6")


def test_empty_voice_command_is_rejected():
    assert not is_english_command_text("   ")
