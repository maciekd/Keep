"""The signature readers and the signature table.

Two kinds of test: the readers on sample texts, and the real texts — res/prompt.md and every
toolkit — held to the canon: every documented command declares its parameters by name, every
example is a plain call with strings in double quotes, and no example passes more values than
the declaration names.
"""
from types import SimpleNamespace

import pytest

from assistant.commands import Toolkits
from assistant.commands.Commands import Commands
from assistant.commands.CommandParser import CommandParser
from assistant.commands.formats import Signatures
from assistant.commands.formats.Signatures import Call, Declaration, Example, Parameter

NAMES = ["turn_on", "dim", "load_toolkit", "save_watch", "restart_server", "browse_eval", "latest_email"]

SAMPLE = """\
Lights: `turn_on(light_name: str)` and `dim(light_name: str, dim_percentage: int)`, e.g. `dim("Kitchen lamp", 1)`.
Load toolkits with `load_toolkit(name: str, ...)` — several at once: `load_toolkit("email", "web")`.
`latest_email(folder: str = "", unread_only: bool = false)` lists mail; `restart_server()` restarts.
Mention `turn_on` without parentheses. A `word(x)` that is not a command, and `turn_on(Kitchen lamp)` bare.
Notes go as a block: `save_watch("sofa, delivery", \"\"\"two
lines\"\"\", "2026-11-08")`. JavaScript is opaque: `browse_eval(document.title)`.
"""


@pytest.mark.unit
class TestReaders:

    def test_finder_yields_every_backticked_call_and_nothing_else(self):
        found = [(c.name, c.inner) for c in Signatures.calls(SAMPLE)]
        assert ("turn_on", "light_name: str") in found
        assert ("word", "x") in found                      # the finder knows no command names
        assert ("restart_server", "") in found
        assert all(name != "turn_on" or inner != "" for name, inner in found), "a plain mention is not a call"
        assert len([c for c in Signatures.calls(SAMPLE) if c.name == "save_watch"]) == 1, "a call may span lines"

    def test_declaration_carries_names_types_defaults_and_the_variadic(self):
        decl = Signatures.read_declaration(Call("latest_email", 'folder: str = "", unread_only: bool = false', 0, 0))
        assert decl == Declaration("latest_email", (Parameter("folder", "str", '""'),
                                                    Parameter("unread_only", "bool", "false")), False)
        assert decl.names == ["folder", "unread_only"]
        variadic = Signatures.read_declaration(Call("load_toolkit", "name: str, ...", 0, 0))
        assert variadic.names == ["name"] and variadic.variadic
        assert Signatures.read_declaration(Call("show_history", "from: YYYY-MM-DD HH:MM, to: HH:MM", 0, 0)).names == ["from", "to"]

    @pytest.mark.parametrize("inner", ["", '"Kitchen lamp"', "Kitchen lamp", "name", "..."])
    def test_what_is_not_a_declaration(self, inner):
        assert Signatures.read_declaration(Call("turn_on", inner, 0, 0)) is None

    def test_example_is_read_by_the_runtime_parser(self):
        example = Signatures.read_example(Call("save_watch", '"sofa, delivery", """two\nlines""", "2026-11-08"', 0, 0))
        assert example.problem is None
        assert example.values == ["sofa, delivery", "two\nlines", "2026-11-08"]
        assert example.values == CommandParser(["save_watch"]).extract_all_commands(
            'save_watch("sofa, delivery", """two\nlines""", "2026-11-08")')[0][1]

    def test_numbers_and_booleans_are_bare_and_a_quote_is_escaped(self):
        example = Signatures.read_example(Call("dim", '"Kitchen \\"lamp\\"", 1, true, -2.5', 0, 0))
        assert example.problem is None
        assert example.values == ['Kitchen "lamp"', "1", "true", "-2.5"]
        assert Signatures.read_example(Call("restart_server", "", 0, 0)) == Example("restart_server", [], None)

    @pytest.mark.parametrize("inner", ["Kitchen lamp", "'Kitchen lamp'", "`Kitchen lamp`", '"a" b', '"a",',
                                       "name", "...", 'light_name: "x"', '"unterminated'])
    def test_what_is_not_canonical(self, inner):
        assert Signatures.read_example(Call("turn_on", inner, 0, 0)).problem is not None

    def test_an_opaque_payload_is_one_value_taken_verbatim(self):
        example = Signatures.read_example(Call("browse_eval", "document.querySelector('a, b').href", 0, 0))
        assert example.problem is None
        assert example.values == ["document.querySelector('a, b').href"]

    def test_scan_sorts_the_calls_of_known_commands(self):
        scan = Signatures.scan(SAMPLE, NAMES)
        assert [d.command for d in scan.declarations] == ["turn_on", "dim", "load_toolkit", "latest_email"]
        assert {e.command for e in scan.examples} == {"dim", "load_toolkit", "restart_server", "turn_on",
                                                       "save_watch", "browse_eval"}
        assert "word" not in {e.command for e in scan.examples}
        assert [e.problem is None for e in scan.examples if e.command == "turn_on"] == [False]

    def test_table_merges_texts_and_a_bare_call_declares_a_parameterless_command(self):
        table = Signatures.build_table({"a": SAMPLE, "b": "`turn_on()` again and `caffeinate()`"}, NAMES + ["caffeinate"])
        assert table["dim"].names == ["light_name", "dim_percentage"]
        assert table["restart_server"].params == () and table["caffeinate"].params == ()
        assert table["turn_on"].names == ["light_name"], "a bare call never overrides a typed declaration"
        assert "save_watch" not in table, "an example alone declares nothing"

    def test_two_declarations_that_differ_are_a_conflict_and_the_first_wins(self):
        texts = {"a": "`dim(light_name: str, dim_percentage: int)`", "b": "`dim(name: str, percentage: int)`",
                 "c": "`dim(light_name: str, dim_percentage: int)`"}
        assert Signatures.conflicts(texts, NAMES) == [("dim", "a", "b")]
        assert Signatures.build_table(texts, NAMES)["dim"].names == ["light_name", "dim_percentage"]


def _documented() -> set[str]:
    return {c for t in Toolkits.TOOLKITS for c in t.offered_commands()} | set(Toolkits.CORE_COMMANDS)


@pytest.mark.unit
class TestRealTexts:
    """res/prompt.md and every toolkit file, as the rewriter of step 3 will read them."""

    def test_every_documented_command_declares_its_parameters(self):
        table = Signatures.build_table(Signatures.texts(), Commands.command_names())
        missing = sorted(_documented() - set(table))
        assert missing == [], f"no declaration (`name(param: type, …)`, or `name()` for one without parameters) for: {missing}"

    def test_every_declared_name_is_a_command(self):
        names = set(Commands.command_names())
        for label, text in Signatures.texts().items():
            for call in Signatures.calls(text):
                if Signatures.read_declaration(call) is not None:
                    assert call.name in names, f"{label} declares `{call.name}(…)`, which is no command"

    def test_no_command_is_declared_twice_with_different_parameters(self):
        assert Signatures.conflicts(Signatures.texts(), Commands.command_names()) == []

    def test_a_toolkit_outside_the_checkout_is_read_under_its_full_path(self, tmp_path, monkeypatch):
        """A local integration's toolkit (extra_integration_dirs) is not under the repository, so
        its label cannot be relative to it (#157 — the boot died on the first prompt)."""
        path = tmp_path / "demo" / "toolkit.md"
        local = SimpleNamespace(path=path, text=lambda: "`dim(light_name: str, dim_percentage: int)`")
        monkeypatch.setattr(Toolkits, "TOOLKITS", [*Toolkits.TOOLKITS, local])

        texts = Signatures.texts()
        assert texts[str(path)] == local.text()
        assert "res/prompt.md" in texts

    def test_every_example_parses_is_canonical_and_fits_its_declaration(self):
        names = Commands.command_names()
        texts = Signatures.texts()
        table = Signatures.build_table(texts, names)
        problems = []
        for label, text in texts.items():
            for example in Signatures.scan(text, names).examples:
                if example.problem:
                    problems.append(f"{label}: `{example.command}(…)` {example.problem}")
                    continue
                declaration = table.get(example.command)
                if declaration is None:
                    problems.append(f"{label}: `{example.command}(…)` has no declaration to pair its values with")
                elif len(example.values) > len(declaration.params) and not declaration.variadic:
                    problems.append(f"{label}: `{example.command}(…)` passes {len(example.values)} values, "
                                    f"{declaration.names} declared")
        assert problems == [], "\n".join(problems)

    def test_the_prompt_teaches_the_quoted_form(self):
        """The COMMAND SYSTEM prose is the Keep format's instructions ({COMMAND_SYSTEM}) since step 3."""
        from assistant.commands.formats.KeepFormat import KeepFormat
        prompt = Signatures.texts()["res/prompt.md"]
        instructions = KeepFormat([]).instructions()
        assert "{COMMAND_SYSTEM}" in prompt
        assert 'turn_on("Kitchen lamp")' in instructions
        assert "turn_on(Kitchen lamp)" not in prompt and "turn_on(Kitchen lamp)" not in instructions

    def test_the_cached_table_is_the_real_one(self):
        Signatures.reset()
        table = Signatures.table()
        assert table is Signatures.table()
        assert table["dim"] == ["light_name", "dim_percentage"]
        assert table["load_toolkit"] == ["name"]
        assert table["restart_server"] == []
