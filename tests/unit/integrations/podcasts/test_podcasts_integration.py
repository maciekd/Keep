import json

import pytest

from integrations.podcasts.PodcastsIntegration import PodcastsIntegration
from integrations.podcasts.PodcastsPipeline import PodcastsPipeline


@pytest.fixture
def integration(config):
    return PodcastsIntegration(config, PodcastsPipeline(config, notify=lambda text: None))


def finished(show_dir, slug="moonshots-412"):
    folder = show_dir / slug
    folder.mkdir()
    (folder / "tldr.md").write_text("- Robots juggle now [0:42]")
    (folder / "manifest.json").write_text(json.dumps({"slug": slug, "title": "Robots", "finished_at": 2e9}))


def test_the_episode_list_names_each_slug_and_its_artifacts(integration, show_dir):
    finished(show_dir)
    (show_dir / "moonshots-300.md").write_text('---\nepisode: "Old"\n---\n')
    lines = integration.list_episodes(5).split("\n")
    assert lines[0].startswith("- moonshots-412") and "artifacts: tldr" in lines[0]
    assert "older pipeline" in lines[1]
    assert integration.list_episodes(1).count("\n") == 0


def test_an_artifact_is_read_whole_and_a_missing_one_is_named(integration, show_dir):
    finished(show_dir)
    assert integration.load_artifact("Moonshots-412", "tldr") == "- Robots juggle now [0:42]"
    assert "has tldr, not 'explainer'" in integration.load_artifact("moonshots-412", "explainer")
    assert "no episode" in integration.load_artifact("moonshots-1", "tldr")


def test_a_missing_output_folder_is_unhealthy(integration, show_dir):
    show_dir.rmdir()
    with pytest.raises(RuntimeError, match="output folder"):
        integration.check_health()
