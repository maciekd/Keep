"""Pin invariants of the production prompt template (`res/prompt.md`).

The prompt template is plain markdown but encodes load-bearing decisions —
notably where the Anthropic prompt-cache breakpoints sit, which directly
controls how much of the ~80KB system prompt gets cached on every voice query.
The template is split stable → volatile by up to MAX_SYSTEM_CACHE_BLOCKS
`<<<CACHE_BREAKPOINT>>>` markers (see ClaudeLLM._build_cached_system): each
marker ends a cached block, so churn in a later segment leaves the earlier
breakpoints' cache reads intact. Moving or removing markers silently changes
the per-request bill, so the placement is asserted here.
"""

import os

import pytest

# AIServer/tests/unit/test_prompt_template.py → AIServer/res/prompt.md
_AISERVER_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROMPT_MD = os.path.join(_AISERVER_DIR, "res", "prompt.md")

CACHE_BREAKPOINT_MARKER = "<<<CACHE_BREAKPOINT>>>"


@pytest.fixture(scope="module")
def prompt_text():
    with open(_PROMPT_MD, "r", encoding="utf-8") as f:
        return f.read()


@pytest.mark.unit
def test_cache_breakpoint_count_within_claude_limit(prompt_text):
    """ClaudeLLM turns each marker into a cached system block, up to
    MAX_SYSTEM_CACHE_BLOCKS (Anthropic allows 4 breakpoints per request; one is
    reserved for the message-history breakpoint). Markers beyond the limit are
    silently demoted into the uncached tail — so adding a marker past the limit
    doesn't buy any caching and must fail here instead."""
    from assistant.llm.ClaudeLLM import ClaudeLLM

    count = prompt_text.count(CACHE_BREAKPOINT_MARKER)
    assert 1 <= count <= ClaudeLLM.MAX_SYSTEM_CACHE_BLOCKS, (
        f"expected 1..{ClaudeLLM.MAX_SYSTEM_CACHE_BLOCKS} {CACHE_BREAKPOINT_MARKER} "
        f"markers, found {count}"
    )


@pytest.mark.unit
def test_final_cache_breakpoint_sits_after_recent_messages_header(prompt_text):
    """The RECENT MESSAGES header and the short-term-memory note below it are
    static (abceaebb added the daily-reset note). They should ride inside the
    cached prefix too — the final breakpoint must sit at the END of the static
    body, not before this section."""
    recent_idx = prompt_text.find("# RECENT MESSAGES")
    last_breakpoint_idx = prompt_text.rfind(CACHE_BREAKPOINT_MARKER)
    assert recent_idx >= 0, "# RECENT MESSAGES header missing from prompt.md"
    assert last_breakpoint_idx > recent_idx, (
        "the final cache breakpoint must sit AFTER the RECENT MESSAGES static "
        "header so the static memory-reset instruction is inside the cached prefix"
    )


@pytest.mark.unit
def test_all_cache_breakpoints_sit_before_thinking_sentinel(prompt_text):
    """`[THINKING]` is a runtime-substituted sentinel — its content (the
    extended-thinking instruction or nothing) varies per request. It must
    therefore sit OUTSIDE every cached block, i.e. after the last breakpoint."""
    thinking_idx = prompt_text.find("[THINKING]")
    last_breakpoint_idx = prompt_text.rfind(CACHE_BREAKPOINT_MARKER)
    assert thinking_idx >= 0, "[THINKING] sentinel missing from prompt.md"
    assert last_breakpoint_idx < thinking_idx, (
        "[THINKING] is substituted per-request and must live in the volatile "
        "tail (after the last cache breakpoint)"
    )


# Backtick-quoted `name(` mentions in the prompt that are not commands the parser can run.
_NOT_A_COMMAND = {"command"}   # the generic `command(argument)` syntax example


def _advertised_commands(text: str) -> set:
    import re
    return set(re.findall(r'`([a-z_]+)\(', text)) - _NOT_A_COMMAND


@pytest.mark.unit
def test_every_command_the_prompt_advertises_exists(prompt_text):
    """A command renamed in Commands.py but not in the prompt (or the other way round)
    costs a silent 'Unknown command' on every use — the LLM keeps calling the old name."""
    from assistant.commands.Commands import Commands

    missing = sorted(_advertised_commands(prompt_text) - set(Commands.command_names()))
    assert not missing, f"res/prompt.md advertises commands with no cmd_ method: {missing}"


# --- toolkits (plans/TOOLKITS_PLAN.md) --------------------------------------------------------

def _signature_lines(text: str) -> set:
    """Commands documented with a signature line of their own (`name(args)` at line start),
    as opposed to a mention in prose or an example."""
    import re
    return set(re.findall(r'^`([a-z_]+)\(', text, flags=re.M)) - _NOT_A_COMMAND


@pytest.mark.unit
def test_prompt_carries_the_toolkit_placeholders(prompt_text):
    assert "{TOOLKITS}" in prompt_text, "res/prompt.md lacks the {TOOLKITS} index placeholder"
    assert "`load_toolkit(" in prompt_text


@pytest.mark.unit
def test_prompt_carries_the_destinations_placeholder_beside_post_message(prompt_text):
    """Contract with PromptManager._substitute: the command block lists where post_message can
    reach through {DESTINATIONS}, filled from assistant.Destinations."""
    assert "{DESTINATIONS}" in prompt_text
    assert prompt_text.index("`post_message(") < prompt_text.index("{DESTINATIONS}")


@pytest.mark.unit
def test_carried_sections_have_no_placeholder_in_the_template(prompt_text):
    """A section a toolkit carries reaches the LLM through load_toolkit; a placeholder in
    res/prompt.md would put the data in every prompt again."""
    from assistant.commands import Toolkits
    for key in Toolkits.CARRIED_SECTIONS:
        assert "{" + key.upper() + "}" not in prompt_text, f"res/prompt.md still carries {{{key.upper()}}}"


@pytest.mark.unit
def test_only_core_commands_are_documented_in_the_template(prompt_text):
    """A loadable toolkit's reference lives in its file; a signature line for one of its
    commands in the template is a second copy that will drift."""
    from assistant.commands import Toolkits
    stray = sorted(_signature_lines(prompt_text) - set(Toolkits.CORE_COMMANDS))
    assert not stray, f"res/prompt.md documents commands that belong to a loadable toolkit: {stray}"
