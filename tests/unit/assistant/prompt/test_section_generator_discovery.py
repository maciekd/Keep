"""The prompt sections an integration folder brings: `*SectionGenerator` modules are found by
discovery, PromptManager runs them beside the core's under the key each declares, and a
section no template placeholder names reaches the LLM carried by the folder's toolkit."""
import importlib
import sys
from unittest.mock import Mock, patch

import pytest

from assistant.commands import Toolkits
from assistant.commands.Commands import Commands
from assistant.prompt import PromptSectionGenerator as SectionGenerators
from assistant.prompt.PromptManager import PromptManager
from assistant.prompt.PromptSectionGenerator import PromptSectionGenerator

GENERATOR = '''from assistant.prompt.PromptSectionGenerator import PromptSectionGenerator


class {name}SectionGenerator(PromptSectionGenerator):
    """{name} state for the prompt."""
    key = "{key}"

    def __init__(self, prompt_manager):
        super().__init__(prompt_manager, "{name}State")
        self.content = "{content}"

    def _fetch_content(self) -> str:
        return self.content
'''

LAMP_TOOLKIT = '''---
name: lamp
summary: "the reading lamp: switch it and see whether it is on"
commands: [lamp_on]
sections: [lamp]
---
`lamp_on()` switches the reading lamp on.
'''


@pytest.fixture
def folders(tmp_path, monkeypatch):
    """A throwaway `integrations`-like package: `make(folder, module, source)` adds a module to it."""
    package = f"folders_{tmp_path.name.replace('-', '_')}"
    root = tmp_path / package
    root.mkdir()
    (root / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))

    def make(folder: str, module: str, source: str):
        (root / folder).mkdir(exist_ok=True)
        (root / folder / "__init__.py").write_text("")
        (root / folder / f"{module}.py").write_text(source)

    make.root = root
    make.load = lambda: importlib.import_module(package)
    yield make
    for name in [name for name in sys.modules if name == package or name.startswith(package + ".")]:
        del sys.modules[name]


@pytest.fixture
def template(tmp_path, monkeypatch):
    """A working directory whose prompt template places the memory bank and the kettle section."""
    (tmp_path / "res").mkdir()
    (tmp_path / "res" / "prompt.md").write_text("Header\n{MEMORY}\n{KETTLE}\nFooter\n")
    monkeypatch.chdir(tmp_path)


def manager(generators) -> PromptManager:
    with patch.object(PromptSectionGenerator, "start", lambda self: None):
        return PromptManager(section_generators=generators)


def test_only_section_generator_modules_are_imported_and_a_broken_one_is_left_out(folders):
    folders("kettle", "KettleSectionGenerator", GENERATOR.format(name="Kettle", key="kettle", content="cold"))
    folders("kettle", "KettleIntegration", GENERATOR.format(name="Misplaced", key="misplaced", content="x"))
    folders("radio", "RadioSectionGenerator", "import a_package_nobody_installed\n")
    package = folders.load()

    found = SectionGenerators.find_in(package)

    assert [cls.__name__ for cls in found] == ["KettleSectionGenerator"]
    assert f"{package.__name__}.kettle.KettleIntegration" not in sys.modules


def test_a_folder_whose_package_raises_costs_only_itself(folders):
    """pkgutil imports a folder's package to walk it; one raising other than an ImportError
    ends the walk without the error hook. It is logged and the folders after it are found."""
    folders("broken", "BrokenSectionGenerator", "")
    (folders.root / "broken" / "__init__.py").write_text("raise RuntimeError('a folder whose package itself fails')\n")
    folders("kettle", "KettleSectionGenerator", GENERATOR.format(name="Kettle", key="kettle", content="cold"))
    package = folders.load()

    with patch.object(SectionGenerators, "error") as error:
        found = SectionGenerators.find_in(package)

    assert [cls.__name__ for cls in found] == ["KettleSectionGenerator"]
    assert any(f"{package.__name__}.broken" in call.args[0] for call in error.call_args_list)


def test_a_folder_section_stands_in_the_prompt_where_its_placeholder_is(folders, template):
    folders("kettle", "KettleSectionGenerator", GENERATOR.format(name="Kettle", key="kettle", content="the kettle is cold"))

    pm = manager(SectionGenerators.find_in(folders.load()))

    assert "Header\n" in pm.prompt and "\nthe kettle is cold\nFooter" in pm.prompt
    assert list(pm.sections)[-1] == "kettle", "a section the pinned order does not know comes after the known ones"
    assert list(pm.sections)[:5] == list(PromptManager.SECTION_ORDER[:5])


def test_a_key_with_two_owners_is_an_error_naming_both(folders, template):
    folders("kettle", "KettleSectionGenerator", GENERATOR.format(name="Kettle", key="vault", content="x"))
    kettle, = SectionGenerators.find_in(folders.load())

    with pytest.raises(ValueError) as raised:
        manager([kettle])

    assert "KettleSectionGenerator" in str(raised.value) and "VaultSectionGenerator" in str(raised.value)


def test_a_key_a_gated_core_section_owns_is_an_error_naming_the_core_owner(folders, template):
    folders("kettle", "KettleSectionGenerator", GENERATOR.format(name="Kettle", key="home", content="x"))
    kettle, = SectionGenerators.find_in(folders.load())

    with pytest.raises(ValueError) as raised:
        manager([kettle])

    assert "KettleSectionGenerator" in str(raised.value) and "HomeSectionGenerator" in str(raised.value)


def test_a_generator_without_a_key_or_with_a_failing_constructor_costs_only_itself(folders, template):
    folders("kettle", "KettleSectionGenerator", GENERATOR.format(name="Kettle", key="kettle", content="cold"))
    folders("radio", "RadioSectionGenerator", GENERATOR.format(name="Radio", key="", content="off"))
    folders("toaster", "ToasterSectionGenerator",
            GENERATOR.format(name="Toaster", key="toaster", content="idle").replace(
                'self.content = "idle"', 'raise RuntimeError("no toaster on the network")'))

    pm = manager(SectionGenerators.find_in(folders.load()))

    assert "kettle" in pm.sections and "toaster" not in pm.sections and "" not in pm.sections
    assert "memory" in pm.sections and "home" not in pm.sections and "home" in pm.section_classes()


def test_load_toolkit_sends_a_folders_carried_section_that_the_prompt_never_holds(folders, template, monkeypatch):
    """The route a folder takes without a template edit: toolkit.md says `sections: [lamp]`."""
    folders("lamp", "LampSectionGenerator", GENERATOR.format(name="Lamp", key="lamp", content="The reading lamp is on."))
    (folders.root / "lamp" / "toolkit.md").write_text(LAMP_TOOLKIT)
    toolkits = Toolkits.assemble(Toolkits.CORE_TOOLKITS, Toolkits.discover(folders.root), Toolkits.INDEX_ORDER)
    monkeypatch.setattr(Toolkits, "TOOLKITS", toolkits)
    monkeypatch.setattr(Toolkits, "_BY_NAME", {t.name: t for t in toolkits})
    monkeypatch.setattr(Toolkits, "CARRIED_SECTIONS", {key: t.name for t in toolkits for key in t.sections})
    pm = manager(SectionGenerators.find_in(folders.load()))
    commands = Commands(Mock(command_reply=None, command_reply_contains_error=False), command_sets=())

    with patch.object(Commands, "_prompt_manager", staticmethod(lambda: pm)):
        first = commands._run_load_toolkit("lamp")
        pm.sections["lamp"].content = "The reading lamp is off."
        kettle = commands._run_load_toolkit("web")

    reference, _, data = first.partition("switches the reading lamp on.")
    assert reference.startswith("# Toolkit: lamp\n") and "sections:" not in first
    assert data.rstrip().endswith("The reading lamp is on.")
    assert "reading lamp is" not in pm.prompt and "{LAMP}" not in pm.prompt
    assert "reading lamp" not in kettle


def test_a_section_nothing_places_or_carries_is_reported_and_harms_nothing(folders, template):
    folders("radio", "RadioSectionGenerator", GENERATOR.format(name="Radio", key="radio", content="the radio plays jazz"))

    with patch("assistant.prompt.PromptManager.error") as reported:
        pm = manager(SectionGenerators.find_in(folders.load()))

    assert "radio" in pm.sections and "plays jazz" not in pm.prompt and pm.prompt.startswith("Header\n")
    assert any("radio" in str(call.args[0]) for call in reported.call_args_list)


def test_every_generator_file_in_this_repository_is_discovered_with_a_pinned_key():
    from common.Paths import INTEGRATIONS_DIR
    on_disk = {path.stem for path in INTEGRATIONS_DIR.glob(f"*/*{SectionGenerators.MODULE_SUFFIX}.py")}
    discovered = SectionGenerators.find_in(importlib.import_module("integrations"))

    assert {cls.__module__.rsplit(".", 1)[-1] for cls in discovered} == on_disk
    keys = [cls.key for cls in discovered]
    assert len(set(keys)) == len(keys) and set(keys) <= set(PromptManager.SECTION_ORDER)
