"""The example as several files: the root config.example.yaml and the fragments in the
integration folders. Every tree here is a temporary one with fictional sections."""
import re

import pytest

from common import ConfigSchema
from common.Paths import AISERVER_DIR, INTEGRATIONS_DIR

FOLDERS = INTEGRATIONS_DIR.relative_to(AISERVER_DIR)

LAMP = """\
# The lamp in the hall. Ask the user which room it lights.
lamp:
  brightness: 40
  rooms: [hall]
"""
KETTLE = """\
kettle:
  litres: 1.5
"""


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """Writes a root example and its fragments under tmp_path; returns (write_root, write_fragment, errors)."""
    errors = []
    monkeypatch.setattr(ConfigSchema, "EXAMPLE_PATH", tmp_path / "config.example.yaml")
    monkeypatch.setattr(ConfigSchema, "error", errors.append)

    def write_root(text: str):
        (tmp_path / "config.example.yaml").write_text(text)
        ConfigSchema.reset()

    def write_fragment(folder: str, text: str) -> str:
        path = tmp_path / FOLDERS / folder / ConfigSchema.FRAGMENT_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        ConfigSchema.reset()
        return f"{FOLDERS.as_posix()}/{folder}/{ConfigSchema.FRAGMENT_NAME}"

    yield write_root, write_fragment, errors
    ConfigSchema.reset()


def test_a_pointed_fragment_stands_where_its_pointer_is(tree):
    write_root, write_fragment, errors = tree
    lamp, kettle = write_fragment("lamp", LAMP), write_fragment("kettle", KETTLE)
    write_root(f"api_keys:\n  openai: KEY\n\n# >> lamp: {lamp}\n\nvault:\n  key: CHANGE_ME\n\n# >> kettle: {kettle}\n")

    assert ConfigSchema.sections() == ["api_keys", "lamp", "vault", "kettle"]
    assert ConfigSchema.section_text("lamp") == LAMP.rstrip("\n")
    assert ConfigSchema.leaf_type("lamp.brightness") is int and ConfigSchema.is_scalar_list("lamp.rooms")
    assert ConfigSchema.leaves("kettle") == ["kettle.litres"]
    assert ">>" not in ConfigSchema.section_text("vault") and ">>" not in ConfigSchema.section_text("api_keys")
    assert errors == []


def test_a_fragment_no_pointer_names_follows_the_root_in_folder_order(tree):
    write_root, write_fragment, errors = tree
    lamp = write_fragment("lamp", LAMP)
    write_fragment("b_shed", "anchor:\n  kilos: 12\n")
    write_fragment("a_shed", "zeppelin:\n  metres: 245\n")
    write_root(f"# >> lamp: {lamp}\n\nvault:\n  key: CHANGE_ME\n")

    assert ConfigSchema.sections() == ["lamp", "vault", "zeppelin", "anchor"]
    assert ConfigSchema.section_text("vault") == "vault:\n  key: CHANGE_ME"
    assert ConfigSchema.example_value("zeppelin.metres") == 245
    assert errors == []


def test_a_fragment_redefining_a_section_is_left_out_and_both_files_are_named(tree):
    write_root, write_fragment, errors = tree
    lamp = write_fragment("lamp", LAMP)
    write_fragment("lantern", "lamp:\n  brightness: 99\n")
    write_fragment("urn", "vault:\n  key: OTHER\nurn:\n  litres: 9\n")
    write_root(f"# >> lamp: {lamp}\n\nvault:\n  key: CHANGE_ME\n")

    assert ConfigSchema.sections() == ["lamp", "vault"]
    assert ConfigSchema.example_value("lamp.brightness") == 40
    assert ConfigSchema.example_value("vault.key") == "CHANGE_ME"
    assert len(errors) == 2
    by_folder = {folder: message for message in errors for folder in ("lantern", "urn") if f"/{folder}/" in message}
    assert f"/lamp/{ConfigSchema.FRAGMENT_NAME}" in by_folder["lantern"]
    assert str(ConfigSchema.EXAMPLE_PATH) in by_folder["urn"]


@pytest.mark.parametrize("broken", ["lamp: [unclosed\n", "# only a note about lamps\n", "- lamp\n- kettle\n"],
                         ids=["not yaml", "no section", "a list"])
def test_a_malformed_fragment_is_left_out_and_the_rest_stands(tree, broken):
    write_root, write_fragment, errors = tree
    lamp, kettle = write_fragment("lamp", broken), write_fragment("kettle", KETTLE)
    write_fragment("radio", broken)
    write_root(f"# >> lamp: {lamp}\n\nvault:\n  key: CHANGE_ME\n\n# >> kettle: {kettle}\n")

    assert ConfigSchema.sections() == ["vault", "kettle"]
    assert ConfigSchema.leaves("kettle") == ["kettle.litres"]
    assert sorted(re.search(r"/(lamp|radio)/", message).group(1) for message in errors) == ["lamp", "radio"]


def test_a_pointer_to_a_missing_fragment_costs_only_that_section(tree):
    write_root, write_fragment, errors = tree
    kettle = write_fragment("kettle", KETTLE)
    write_root(f"# >> lamp: {FOLDERS.as_posix()}/lamp/{ConfigSchema.FRAGMENT_NAME}\n\n# >> kettle: {kettle}\n")

    assert ConfigSchema.sections() == ["kettle"]
    assert len(errors) == 1 and "/lamp/" in errors[0]


def test_every_fragment_of_this_repository_is_pointed_at_from_the_root_under_its_own_section_name():
    """A fragment the root does not point at would still load, but at the end of the list:
    the order of config_help is part of what the assistant sees."""
    ConfigSchema.reset()
    try:
        root_lines = ConfigSchema.EXAMPLE_PATH.read_text().splitlines()
        pointers = {m.group(2): m.group(1) for m in map(ConfigSchema._POINTER_RE.match, root_lines) if m}
        fragments = sorted(INTEGRATIONS_DIR.glob(f"*/{ConfigSchema.FRAGMENT_NAME}"))
        assert fragments, "no fragments found — the glob is wrong"
        for path in fragments:
            relative = path.relative_to(AISERVER_DIR).as_posix()
            assert relative in pointers, f"{relative} has no pointer line in config.example.yaml"
            defined = ConfigSchema._section_names(path.read_text().splitlines())
            assert pointers[relative] in defined, f"the pointer calls {relative} '{pointers[relative]}', it defines {defined}"
            assert set(defined) <= set(ConfigSchema.sections()), f"{relative} was left out of the schema"
        assert len(pointers) == len(fragments), "a pointer names a fragment that is not there"
    finally:
        ConfigSchema.reset()


@pytest.fixture
def real_schema():
    ConfigSchema.reset()
    yield
    ConfigSchema.reset()


@pytest.mark.parametrize("section", ["local_outposts", "watchdog", "imessage_satellite", "photos", "reminders", "homeclaw"])
def test_what_only_a_mac_runs_says_so_at_the_top_of_its_help(real_schema, section):
    """config_help shows the section verbatim: on a Linux server the first thing the assistant
    reads of it must be that it is not for this machine."""
    first = ConfigSchema.section_text(section).lstrip().splitlines()[0]
    assert "macOS only" in first, first


@pytest.mark.parametrize("section", ["imessage", "imessage_chat"])
def test_the_imessage_help_tells_a_linux_server_it_needs_a_satellite(real_schema, section):
    """The Linux first-run brief and the setup toolkit send the assistant here for exactly that."""
    text = ConfigSchema.section_text(section)
    linux = next(line for line in text.splitlines() if "Linux server" in line)
    assert "local" in text[text.index(linux):] and "satellite" in text[text.index(linux):]


def test_the_imessage_help_carries_the_satellites_install_command(real_schema):
    """Contract with install.sh's KEEP_ROLE switch and the readme's install line: the command
    the assistant reads out is the one that installs a satellite."""
    from common import Role
    text = ConfigSchema.section_text("imessage")
    install = (AISERVER_DIR / "readme.md").read_text()
    command = re.search(r"^curl -fsSL \S+/install\.sh \| bash$", install, re.M).group(0)
    assert command.replace("| bash", f"| KEEP_ROLE={Role.IMESSAGE_SATELLITE} bash") in text
