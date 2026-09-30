from thirdeye.detector import target_prompts


def test_target_prompts_add_visual_aliases():
    assert target_prompts("water bottle") == ["water bottle", "bottle", "plastic bottle"]


def test_unknown_target_remains_available():
    assert target_prompts("red stapler") == ["red stapler"]
