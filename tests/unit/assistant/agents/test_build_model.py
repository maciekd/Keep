"""The model a Claude Code run uses: a Fable or an Opus one, whatever config.yaml names."""
import json
from unittest.mock import Mock, patch

import pytest

from assistant.agents import BuildPreflight
from assistant.agents.CodeQualityMaintainer import CodeQualityMaintainer, ClaudeStatus, CLAUDE_PATH, build_model

BUILD_MODELS = ["claude-fable-5-2", "claude-mythos-5-2", "claude-opus-5-5", "opus", "claude-opus-5-5[1m]"]
SMALLER_MODELS = ["claude-sonnet-5-5", "sonnet", "claude-haiku-5-5", "gemini-3.8-flash", None]
LOGGED_IN = ClaudeStatus(CLAUDE_PATH, True, "logged in as m@mlab.test (max)")


def _configured(model):
    return patch("assistant.agents.CodeQualityMaintainer.parse_llm_config", return_value=("ClaudeLLM", model))


@pytest.mark.parametrize("model", BUILD_MODELS)
def test_a_fable_or_an_opus_model_is_the_one_the_runs_use(model):
    with _configured(model):
        assert build_model() == model


@pytest.mark.parametrize("model", SMALLER_MODELS)
def test_a_config_that_names_a_smaller_model_runs_on_opus(model):
    with _configured(model):
        assert "opus" in build_model()


def test_a_run_is_never_started_on_sonnet():
    process = Mock(stdout=iter([]))
    process.stderr.read.return_value = ""
    with _configured("claude-sonnet-5-5"), \
            patch("assistant.agents.CodeQualityMaintainer.claude_status", return_value=LOGGED_IN), \
            patch("assistant.agents.CodeQualityMaintainer.subprocess.Popen", return_value=process) as popen:
        CodeQualityMaintainer._run_claude("fix the ferry times", "label")
    cmd = popen.call_args[0][0]
    assert "opus" in cmd[cmd.index("--model") + 1]
    assert not any("sonnet" in str(part) for part in cmd)


class TestTheCreditsRequest:
    @pytest.fixture(autouse=True)
    def nothing_remembered(self):
        BuildPreflight._credits_passed.clear()
        yield
        BuildPreflight._credits_passed.clear()

    @staticmethod
    def _refused(model):
        answer = Mock(stdout=json.dumps({"type": "result", "is_error": True, "result": "You're out of usage credits."}),
                      stderr="", returncode=1)
        with _configured(model), patch("assistant.agents.BuildPreflight.subprocess.run", return_value=answer) as run:
            return BuildPreflight.claude_credits(LOGGED_IN, 0), run.call_args[0][0]

    def test_a_refused_fable_names_opus_as_the_way_out_and_no_smaller_model(self):
        """On 2026-11-18 the text said "a model it has", and the assistant chose Sonnet."""
        why, _ = self._refused("claude-fable-5-2")
        assert "claude-opus-5-5" in why and "ai.code_quality.model" in why
        assert "a model it has" not in why

    def test_a_refused_opus_is_not_told_to_switch_to_itself(self):
        why, _ = self._refused("claude-opus-5-5")
        assert "ai.code_quality.model" not in why

    def test_the_request_is_sent_for_the_model_the_run_would_use(self):
        _, cmd = self._refused("claude-sonnet-5-5")
        assert "opus" in cmd[cmd.index("--model") + 1]
