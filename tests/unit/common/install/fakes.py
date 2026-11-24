"""A Terminal nobody sits at: what it would ask is answered from a list, and what it says is kept."""
import io

from common.install.Terminal import Terminal


class ScriptedTerminal(Terminal):
    """`answers` are what the user types at successive questions (None: there is no terminal to
    ask on; a list that runs out answers Enter). `prompts` and `secret` record what was asked and
    whether the typing was hidden; `said` is everything printed."""

    def __init__(self, answers=None, log_path: str = "", **kwargs):
        self.answers = None if answers is None else list(answers)
        self.prompts: list[str] = []
        self.secret: list[bool] = []
        super().__init__(log_path=log_path, out=io.StringIO(), env={}, **kwargs)

    def _has_terminal(self) -> bool:
        return self.answers is not None

    def _read(self, prompt: str, secret: bool) -> str:
        self.prompts.append(prompt)
        self.secret.append(secret)
        return self.answers.pop(0) if self.answers else ""

    @property
    def said(self) -> str:
        return self.out.getvalue()
