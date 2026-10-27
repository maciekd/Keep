import json
from datetime import datetime
from unittest.mock import patch

import pytest

from integrations.podcasts import fetch_and_transcribe as source
from integrations.podcasts.PodcastsPipeline import MAX_FAILED_ATTEMPTS, PodcastsPipeline, describe_timings
from integrations.podcasts.ReportLLM import Completion

from .conftest import write_bundle


class EchoLLM:
    label = "openai:gpt-test"

    def complete(self, system, user, **kwargs):
        return Completion(text="## 1. An idea\n- **Source:** [2:10] Fusion", seconds=0.0)


@pytest.fixture
def messages():
    return []


@pytest.fixture
def pipeline(config, messages):
    return PodcastsPipeline(config, notify=messages.append, llm_factory=lambda c: EchoLLM())


def fake_fetch_next(found_slug="moonshots-412", video_id="vid412"):
    def fetch_next(playlist_url, is_processed, out_root, prefix, progress, on_found):
        if is_processed(video_id, found_slug):
            return None
        on_found(source.Candidate(video_id, "Robots Learn To Juggle | EP #412", found_slug))
        return write_bundle(out_root, found_slug, video_id)
    return fetch_next


def test_no_check_before_the_check_time_then_one_an_hour(pipeline):
    assert not pipeline.due(datetime(2026, 10, 20, 4, 59))
    assert pipeline.due(datetime(2026, 10, 20, 5, 0))
    pipeline.last_check_at = datetime(2026, 10, 20, 5, 0).timestamp()
    assert not pipeline.due(datetime(2026, 10, 20, 5, 59))
    assert pipeline.due(datetime(2026, 10, 20, 6, 0))


def test_a_new_episode_gets_its_report_is_marked_processed_and_the_owner_hears_twice(pipeline, messages, show_dir):
    with patch.object(source, "fetch_next", side_effect=fake_fetch_next()):
        done = pipeline.run_check()

    assert [m["slug"] for m in done] == ["moonshots-412"]
    assert (show_dir / "moonshots-412" / "manifest.json").exists()
    assert "vid412" in pipeline.load_state()["processed"]
    assert len(messages) == 2 and "Robots Learn To Juggle" in messages[0] and "report ready" in messages[1]
    assert pipeline.status()["slug"] == "moonshots-412"


def test_an_episode_with_a_report_of_the_older_pipeline_is_not_redone(pipeline, messages, show_dir):
    (show_dir / "moonshots-412.md").write_text("---\nepisode: \"Robots\"\n---\n")
    with patch.object(source, "fetch_next", side_effect=fake_fetch_next()):
        assert pipeline.run_check() == []
    assert messages == [] and not (show_dir / "moonshots-412").exists()


def test_a_failing_episode_is_retried_then_given_up_on(pipeline, messages):
    def failing(playlist_url, is_processed, out_root, prefix, progress, on_found):
        if is_processed("vid9", "moonshots-9"):
            return None
        on_found(source.Candidate("vid9", "Broken | EP #9", "moonshots-9"))
        raise OSError("connection reset")

    with patch.object(source, "fetch_next", side_effect=failing):
        for _ in range(MAX_FAILED_ATTEMPTS + 1):
            pipeline.run_check()

    state = pipeline.load_state()
    assert "vid9" in state["processed"] and "vid9" not in state["failed_attempts"]
    assert sum("given up" in m for m in messages) == 1


def test_a_missing_tool_or_api_key_is_said_once_and_nothing_is_fetched(config, values, messages):
    values["podcasts.openai_api_key"] = ""
    pipeline = PodcastsPipeline(config, notify=messages.append)
    with patch.dict("os.environ", {"OPENAI_API_KEY": ""}), patch.object(source, "fetch_next") as fetch:
        pipeline.run_check()
        pipeline.run_check()
    fetch.assert_not_called()
    assert len(messages) == 1 and "API key" in messages[0]


def test_without_yt_dlp_the_check_stops_with_the_reason(pipeline, messages):
    with patch.object(source, "fetch_next", side_effect=source.ToolMissing("yt-dlp is not installed")):
        assert pipeline.run_check() == []
    assert pipeline.last_problem == "yt-dlp is not installed" and messages == ["Podcasts: yt-dlp is not installed"]


def test_a_draft_is_saved_beside_the_ideas(pipeline, show_dir):
    with patch.object(source, "fetch_next", side_effect=fake_fetch_next()):
        pipeline.run_check()
    answer = pipeline.draft("moonshots-412", 1)
    assert (show_dir / "moonshots-412" / "linkedin_draft_1.md").exists() and "An idea" in answer
    assert "failed" in pipeline.draft("moonshots-412", 7)


def test_an_episode_of_the_older_pipeline_cannot_be_regenerated(pipeline, show_dir):
    (show_dir / "moonshots-300.md").write_text("---\nepisode: \"Old\"\n---\n")
    assert "cannot be regenerated" in pipeline.start_regenerate("moonshots-300", "tldr")
    assert "no episode" in pipeline.start_regenerate("moonshots-999", "tldr")


def test_the_timings_read_in_pipeline_order():
    assert describe_timings({"tldr_s": 5, "download_s": 61, "explainer_s": 3700}) == "download 1m01s, explainer 1h01m, TL;DR 5s"
