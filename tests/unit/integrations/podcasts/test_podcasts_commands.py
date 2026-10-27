from unittest.mock import Mock, patch

import pytest

from integrations.podcasts.PodcastsIntegration import PodcastsIntegration


@pytest.fixture
def podcasts():
    with patch.object(PodcastsIntegration, "get_instance", return_value=Mock()) as get_instance:
        yield get_instance.return_value


def ran(commands_instance, command, params):
    """The reply of the work the command handed to the background, or its error."""
    with patch.object(commands_instance, "background_command") as background:
        error = commands_instance.run(command, params)
    if error:
        return error
    function, argument = background.call_args.args
    return function(argument)


def test_the_episodes_default_to_five(commands_instance, podcasts):
    ran(commands_instance, "podcast_episodes", [""])
    podcasts.list_episodes.assert_called_once_with(5)
    assert "number" in ran(commands_instance, "podcast_episodes", ["many"])


def test_an_artifact_is_loaded_by_slug_and_kind(commands_instance, podcasts):
    ran(commands_instance, "load_podcast_artifact", [" moonshots-412", " tldr "])
    podcasts.load_artifact.assert_called_once_with("moonshots-412", "tldr")
    assert "failed" in ran(commands_instance, "load_podcast_artifact", ["moonshots-412"])


def test_a_draft_needs_a_numbered_idea(commands_instance, podcasts):
    ran(commands_instance, "podcast_draft", ["moonshots-412", "2"])
    podcasts.pipeline.draft.assert_called_once_with("moonshots-412", 2)
    assert "failed" in ran(commands_instance, "podcast_draft", ["moonshots-412", "second"])


def test_search_and_regenerate_reach_the_integration(commands_instance, podcasts):
    ran(commands_instance, "podcast_search", ["tokamak"])
    podcasts.search.assert_called_once_with("tokamak")
    ran(commands_instance, "podcast_regenerate", ["moonshots-412", "explainer"])
    podcasts.pipeline.start_regenerate.assert_called_once_with("moonshots-412", "explainer")
    assert "failed" in ran(commands_instance, "podcast_search", [" "])
