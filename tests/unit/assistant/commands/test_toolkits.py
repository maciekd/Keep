"""The toolkit registry and its files (plans/TOOLKITS_PLAN.md): every command has exactly one
home, every file documents exactly the commands it owns, and the rendering the prompt and
load_toolkit share is stable."""
import re
from unittest.mock import patch

import pytest

from assistant.commands import Toolkits
from assistant.commands.Commands import Commands
from assistant.commands.Toolkits import TOOLKITS, Toolkit, ToolkitFileError
from common.Paths import INTEGRATIONS_DIR


def _documented(text: str) -> set[str]:
    return set(re.findall(r"`([a-z_0-9]+)\(", text))


@pytest.mark.unit
def test_every_command_has_exactly_one_home():
    commands = set(Commands(None).list_commands())
    owned = [c for t in TOOLKITS for c in t.commands]
    assert len(owned) == len(set(owned)), "a command is listed in two toolkits"
    assert commands - set(owned) - Toolkits.CORE_COMMANDS - Toolkits.UNDOCUMENTED == set(), \
        "a cmd_* method belongs to no toolkit"
    assert Toolkits.UNDOCUMENTED <= commands and not Toolkits.UNDOCUMENTED & set(owned)
    assert set(owned) - commands == set(), "a toolkit lists a command that does not exist"
    assert Toolkits.CORE_COMMANDS - commands == set()


@pytest.mark.unit
@pytest.mark.parametrize("toolkit", TOOLKITS, ids=[t.name for t in TOOLKITS])
def test_file_documents_its_commands_and_no_others(toolkit: Toolkit):
    assert toolkit.path.exists(), f"{toolkit.path} is missing"
    text = toolkit.text()
    assert len(text) <= Toolkits.MAX_FILE_CHARS
    documented = _documented(text)
    for command in toolkit.commands:
        assert command in documented, f"{toolkit.name}.md does not show `{command}(`"
    foreign = {c for c in documented if Toolkits.owner_of(c) not in (None, toolkit)}
    # A file may point at another toolkit's command in prose or an example (the x toolkit names
    # open_url, location shows a config_set call); it must not carry a second reference for it,
    # which a signature line — a line that is nothing but the backticked call — would be.
    for command in foreign:
        assert not re.search(rf"^`{command}\([^`]*\)`\s*$", text, flags=re.M), \
            f"{toolkit.name}.md documents `{command}` which belongs to {Toolkits.owner_of(command).name}"


@pytest.mark.unit
def test_carried_sections_are_real_sections_with_a_heading():
    from assistant.prompt.PromptManager import PromptManager
    with patch("assistant.prompt.PromptSectionGenerator.PromptSectionGenerator.start"):
        keys = set(PromptManager(test_suite=True).section_classes())   # the gated ones (home) included
    for key, name in Toolkits.CARRIED_SECTIONS.items():
        assert key in keys and Toolkits.by_name(name) is not None
        assert key in Toolkits.SECTION_HEADINGS
    assert Toolkits.render_sections(Toolkits.by_name("email"), {"home": "x"}) == ""
    assert Toolkits.render_sections(Toolkits.by_name("home"), {"home": "  \n"}) == ""
    rendered = Toolkits.render_sections(Toolkits.by_name("home"), {"home": "## LIGHTS\nlamp: off\n\n"})
    assert rendered == "## HOME AUTOMATION DATA\n## LIGHTS\nlamp: off\n"


@pytest.mark.unit
def test_a_carried_section_has_no_heading_of_its_own_so_the_prompt_does_not_carry_it_too():
    """The toolkit brings the data on load_toolkit and the heading comes from SECTION_HEADINGS;
    a generator with a heading would stand in every prompt's data block as well, and the LLM
    would read the data twice — once stale, once current."""
    from assistant.prompt.PromptManager import PromptManager
    with patch("assistant.prompt.PromptSectionGenerator.PromptSectionGenerator.start"):
        classes = PromptManager(test_suite=True).section_classes()
    for key in Toolkits.CARRIED_SECTIONS:
        assert classes[key].heading == "", f"section '{key}' is carried by a toolkit and also has a prompt heading"


@pytest.mark.unit
def test_by_name_is_tolerant():
    assert Toolkits.by_name("Email").name == "email"
    assert Toolkits.by_name('"email"').name == "email"
    assert Toolkits.by_name("the email toolkit").name == "email"
    assert Toolkits.by_name("nope") is None
    assert Toolkits.by_name("") is None


@pytest.mark.unit
def test_index_lists_every_offered_toolkit_once():
    index = Toolkits.render_index()
    for toolkit in TOOLKITS:
        listed = not toolkit.hidden and toolkit.offered_by(None)
        assert index.count(f"`{toolkit.name}`") == (1 if listed else 0), toolkit.name
        assert (toolkit.summary in index) == listed


@pytest.mark.unit
def test_the_voice_toolkit_is_hidden_but_loadable():
    """A text chat must never be offered the voice dialog style; the voice channel's preload
    still finds it by name."""
    assert "voice" not in Toolkits.names()
    assert Toolkits.by_name("voice").hidden
    assert "false_positive()" in Toolkits.render(Toolkits.by_name("voice"))


@pytest.mark.unit
def test_render_carries_the_header_and_substitutes():
    toolkit = Toolkits.by_name("timers")
    rendered = Toolkits.render(toolkit, lambda text: text.replace("{WAKE_INTERVAL_MINUTES}", "7"))
    assert rendered.startswith("# Toolkit: timers\n")
    assert "wake up every 7 minutes" in rendered
    assert "{WAKE_INTERVAL_MINUTES}" not in rendered
    assert Toolkits.loaded_name(rendered) == "timers"
    assert Toolkits.loaded_name("result: \n# Toolkit: timers") is None
    assert Toolkits.loaded_name("# Toolkit: ") is None


# --- toolkits an integration folder owns ---------------------------------------------------

LAMP_BODY = "Switch the lamp:\n\n`lamp_on(room: str)`\n`lamp_off(room: str)`\n"


def _folder_toolkit(root, folder: str, front_matter: str, body: str = LAMP_BODY):
    path = root / folder / Toolkits.FOLDER_TOOLKIT_FILE
    path.parent.mkdir(parents=True)
    path.write_text(front_matter + body, encoding="utf-8")
    return path


def _lamp(root, folder: str = "lamp", name: str = "lamp", commands: str = "lamp_on, lamp_off"):
    return _folder_toolkit(root, folder, f'---\nname: {name}\nsummary: "the lamps: on and off"\n'
                                         f"commands: [{commands}]\n---\n")


@pytest.mark.unit
def test_every_toolkit_file_in_the_integration_folders_is_registered_at_a_pinned_place():
    """`discover` leaves a broken toolkit.md out so a bad folder cannot stop the server; for the
    folders of this repository that must be a failing test instead. And a toolkit of the
    repository has its place in the index pinned, because the index is part of the prompt."""
    assert {t.file for t in TOOLKITS if t.file} == set(INTEGRATIONS_DIR.glob(f"*/{Toolkits.FOLDER_TOOLKIT_FILE}"))
    assert [t.name for t in TOOLKITS] == [name for name in Toolkits.INDEX_ORDER if Toolkits.by_name(name)]
    assert len(TOOLKITS) == len(Toolkits.INDEX_ORDER)


@pytest.mark.unit
def test_a_folder_toolkit_reaches_the_llm_without_its_front_matter(tmp_path):
    toolkit = Toolkits.read_folder_toolkit(_lamp(tmp_path))
    assert (toolkit.name, toolkit.commands, toolkit.sections, toolkit.hidden) == ("lamp", ("lamp_on", "lamp_off"), (), False)
    assert toolkit.summary == "the lamps: on and off"
    assert toolkit.text() == LAMP_BODY
    assert Toolkits.render(toolkit) == f"{Toolkits.HEADER_PREFIX}lamp\n{LAMP_BODY}"


@pytest.mark.unit
def test_a_folder_toolkit_may_carry_sections_and_be_hidden(tmp_path):
    path = _folder_toolkit(tmp_path, "kettle", "---\nname: kettle\nsummary: boil water\ncommands: [boil]\n"
                                               "sections: [kettle_state]\nhidden: true\n---\n")
    toolkit = Toolkits.read_folder_toolkit(path)
    assert toolkit.sections == ("kettle_state",) and toolkit.hidden


@pytest.mark.unit
def test_a_core_reference_that_opens_with_a_rule_keeps_it():
    """Only a folder's toolkit.md has front matter; a res/toolkits file is the reference whole."""
    core = Toolkit("lamp", "the lamps", ("lamp_on",))
    with patch("pathlib.Path.read_text", return_value="---\nnot: front matter\n---\n" + LAMP_BODY):
        assert core.text().startswith("---\nnot: front matter")


@pytest.mark.unit
@pytest.mark.parametrize("front_matter", [
    "",                                                                      # none at all
    "---\nname: lamp\nsummary: [unclosed\ncommands: [lamp_on]\n---\n",      # not YAML
    "---\n- lamp\n---\n",                                                   # not a mapping
    "---\nname: lamp\ncommands: [lamp_on]\n---\n",                          # no summary
    "---\nname: lamp\nsummary: the lamps\ncommands: lamp_on\n---\n",        # commands not a list
    "---\nname: lamp\nsummary: the lamps\ncommands: []\n---\n",             # no command
    "---\nname: lamp\nsummary: the lamps\ncommands: [lamp_on]\nsection: [lamp_state]\n---\n",   # misspelt optional key
    "---\nname: lamp\nsummary: the lamps\ncommands: [lamp_on]\nhidden: maybe\n---\n",
], ids=["no front matter", "not yaml", "not a mapping", "no summary", "commands not a list", "no commands",
        "unknown key", "hidden not a bool"])
def test_a_broken_toolkit_file_is_named_and_left_out_while_its_neighbours_load(tmp_path, front_matter):
    broken = _folder_toolkit(tmp_path, "broken", front_matter)
    _lamp(tmp_path, folder="radio", name="radio", commands="radio_on")
    with pytest.raises(ToolkitFileError) as raised:
        Toolkits.read_folder_toolkit(broken)
    assert str(broken) in str(raised.value)
    assert [t.name for t in Toolkits.discover(tmp_path)] == ["radio"]


@pytest.mark.unit
def test_a_discovered_toolkit_takes_its_pinned_place_and_an_unknown_one_follows_by_name(tmp_path):
    core = (Toolkit("home", "the house", ("turn_on",)), Toolkit("voice", "speaking", ("false_positive",), hidden=True))
    for folder, name, command in (("b", "radio", "radio_on"), ("a", "lamp", "lamp_on"), ("c", "kettle", "boil")):
        _lamp(tmp_path, folder=folder, name=name, commands=command)
    registry = Toolkits.assemble(core, Toolkits.discover(tmp_path), order=("home", "lamp", "voice"))
    assert [t.name for t in registry] == ["home", "lamp", "voice", "kettle", "radio"]


@pytest.mark.unit
def test_a_folder_toolkit_cannot_take_a_name_or_a_command_that_is_owned(tmp_path):
    core = (Toolkit("home", "the house", ("turn_on", "turn_off")),)
    _lamp(tmp_path, folder="a_first", name="lamp", commands="lamp_on")
    _lamp(tmp_path, folder="b_same_name", name="lamp", commands="lamp_dim")
    _lamp(tmp_path, folder="c_core_name", name="home", commands="open_door")
    _lamp(tmp_path, folder="d_core_command", name="switches", commands="turn_off, flip")
    _lamp(tmp_path, folder="e_folder_command", name="bulbs", commands="lamp_on")
    registry = Toolkits.assemble(core, Toolkits.discover(tmp_path), order=())
    assert {t.name: t.commands for t in registry} == {"home": ("turn_on", "turn_off"), "lamp": ("lamp_on",)}
    assert next(t for t in registry if t.name == "lamp").file.parent.name == "a_first"
