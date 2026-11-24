"""The installer's steps that need a Python, in order: for a Keep server the AI model — Claude and
its API key, or a model set up on this Mac — and the config, for an iMessage satellite the config
alone.

`install.sh` hands over what it knows in the environment — `KEEP_DIR`, `KEEP_INSTALL_BRANCH`,
`KEEP_INSTALL_ROLE`, `KEEP_INSTALL_LOG`, `KEEP_INSTALL_TELEMETRY_ID`, and where its step counter
stands (`KEEP_INSTALL_STEP` of `KEEP_INSTALL_STEPS`) — beside the user's own variables (`ANTHROPIC_API_KEY`, `KEEP_AI` …). It reads
back the exit status and config.yaml, and after a failure the log's last two lines."""
import os
import sys
import traceback

from common.install.AiChoice import LOCAL, AiChoice
from common.install.AnthropicKey import AnthropicKey
from common.install.InstallConfig import SATELLITE_ROLE, InstallConfig
from common.install.LocalRunner import LocalModel, LocalRunner
from common.install.Terminal import InstallFailed, Terminal

INTERRUPTED = 130


def run(terminal: Terminal, env: dict) -> None:
    config = InstallConfig(terminal, env)
    if env.get("KEEP_INSTALL_ROLE") == SATELLITE_ROLE:
        terminal.step("Configuration")
        config.write_satellite()
        return
    terminal.step("AI model")
    choice = AiChoice(terminal, env)
    local = set_up_local(terminal, env, choice) if choice.choose() == LOCAL else None
    api_key, workspace = ("", "") if local else AnthropicKey(terminal, env, env["KEEP_DIR"]).obtain()
    terminal.step("Configuration")
    config.write_server(api_key, workspace, local)


def set_up_local(terminal: Terminal, env: dict, choice: AiChoice) -> LocalModel | None:
    """The local model, or None when its setup failed and the install goes on with Claude: a broken
    local setup costs the local option, never the install. Asked for by name (`KEEP_AI=local`),
    its failure is the install's — nobody is there to give a key instead."""
    try:
        return LocalRunner(terminal, env).set_up()
    except InstallFailed as failure:
        if choice.explicit:
            raise
        terminal.log(f"local AI: {failure.reason}")
        terminal.warn(failure.reason)
        for line in failure.notes:
            terminal.note(line)
    except Exception as e:
        if choice.explicit:
            raise
        terminal.log(traceback.format_exc())
        terminal.warn(f"The local setup stopped on an unexpected error ({type(e).__name__}: {e}).")
    terminal.note("Local AI is not set up. The install goes on with Claude; Keep can move you to local AI later.")
    terminal.note()
    return None


def main(env=None) -> int:
    env = os.environ if env is None else env
    terminal = Terminal(log_path=env.get("KEEP_INSTALL_LOG", ""), step=int(env.get("KEEP_INSTALL_STEP") or 0),
                        steps=int(env.get("KEEP_INSTALL_STEPS") or 0), env=env)
    try:
        run(terminal, env)
    except InstallFailed as failure:
        terminal.failed(failure)
        return 1
    except KeyboardInterrupt:
        return INTERRUPTED
    except Exception as e:
        terminal.log(traceback.format_exc())
        terminal.failed(InstallFailed(f'Unexpected error in step "{terminal.step_name}" ({type(e).__name__}: {e}).'))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
