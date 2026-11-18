"""The model a Claude Code run uses: a Fable or an Opus one, whatever config.yaml names."""
from unittest.mock import Mock, patch

import pytest

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

