from thirdeye.repo_guidance_audio import RepoGuidanceAnnouncer
from thirdeye.repo_object_finding import RepoGuidance


def test_repo_announcer_uses_reference_interval_for_changed_guidance():
    announcer = RepoGuidanceAnnouncer(repeat_interval_s=1.5)

    assert announcer.update(RepoGuidance.LEFT, now=0.0) == "Left"
    assert announcer.update(RepoGuidance.RIGHT, now=1.0) is None
    assert announcer.update(RepoGuidance.RIGHT, now=1.5) == "Right"


def test_repo_announcer_repeats_unchanged_guidance_after_double_interval():
    announcer = RepoGuidanceAnnouncer(repeat_interval_s=1.5)

    assert announcer.update(RepoGuidance.LEFT, now=0.0) == "Left"
    assert announcer.update(RepoGuidance.LEFT, now=1.5) is None
    assert announcer.update(RepoGuidance.LEFT, now=3.0) == "Left"
