"""The integration folders (src/integrations/<name>/) keep their sides apart: the file suffix is
the role, a `*Tab.py` being the console's and everything else the assistant's, and core finds a
folder by discovery or through a contract. The two allowlists are the by-name imports still
left — an entry goes when its import does, and none is added. Each wants a contract of its own."""
import ast
import re
from pathlib import Path

import pytest

from common.Paths import AISERVER_DIR, SRC_DIR

CORE_PACKAGES = ("assistant", "common")

# core module -> the integration folders it still imports by name
CORE_IMPORTS_BY_NAME = {
    # mails the nightly report through Gmail; Self Improvement is the integration that switches it on
    "assistant/agents/CodeQualityMaintainer.py": {"gmail", "self_improvement"},
    # the nightly curation reads the calendar's coming events to judge what in the bank is still current
    "assistant/agents/LongTermMemoryCurator.py": {"caldav"},
    # fetch_camera_images asks Arlo and Ring; the mail commands build their providers from Gmail and iCloud Email
    "assistant/commands/Commands.py": {"arlo", "gmail", "icloud_email", "ring"},
    # //wipe logs the Telegram Feed session out on the account before its file is deleted
    "assistant/commands/DebugCommands.py": {"telegram_feed"},
}

# (folder, the folder it imports): the cameras' 2FA codes arrive by mail or iMessage, iMessage
# hands the codes it sees to iCloud and Ring, Photos downloads through the iCloud session
FOLDER_IMPORTS_FOLDER = {
    ("arlo", "gmail"), ("arlo", "icloud_email"),
    ("imessage", "icloud"), ("imessage", "ring"),
    ("photos", "icloud"),
}


IMPORT_STATEMENT = re.compile(r"^[ \t]*(?:from|import)[ \t]+[^\n]*", re.MULTILINE)


def import_statements(text: str) -> str:
    """The import statements of a module as a module of their own, dedented: parsing only
    those is a tenth of parsing the whole tree (WebAccess.py alone is 250 KB). A parenthesised
    or backslash-continued statement is taken whole; a prose line that happens to start with
    the keyword (a docstring's) is dropped as what it is, not a statement."""
    statements = []
    for match in IMPORT_STATEMENT.finditer(text):
        statement, end = match.group().strip(), match.end()
        while end < len(text) and (("(" in statement and ")" not in statement) or statement.endswith("\\")):
            next_end = text.find("\n", end + 1)
            next_end = len(text) if next_end < 0 else next_end
            statement, end = statement + "\n" + text[end + 1:next_end], next_end
        statements.append(statement)
    try:
        ast.parse("\n".join(statements))
    except SyntaxError:
        statements = [statement for statement in statements if _parses(statement)]
    return "\n".join(statements)


def _parses(statement: str) -> bool:
    try:
        ast.parse(statement)
    except SyntaxError:
        return False
    return True


def imported_modules(path: Path, src: Path) -> set[str]:
    """Every module `path` imports, absolute: `from x import y` counts as both `x` and `x.y`
    (y may be a module), a relative import is resolved against the file's package."""
    package = list(path.relative_to(src).with_suffix("").parts[:-1])
    found = set()
    for node in ast.walk(ast.parse(import_statements(path.read_text(encoding="utf-8")))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[:len(package) - (node.level - 1)] if node.level else []
            module = ".".join(base + ([node.module] if node.module else []))
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
    return found


def survey(src: Path) -> dict:
    """What the four rules look at, read from the tree under `src`."""
    found = {"interface_imports": set(), "core_tab_imports": set(), "core_by_name": {}, "folder_to_folder": set()}
    for path in sorted(src.rglob("*.py")):
        rel = path.relative_to(src)
        top = rel.parts[0]
        if top not in CORE_PACKAGES + ("integrations",):
            continue
        text = path.read_text(encoding="utf-8")
        if "integrations" not in text and "interfaces" not in text:
            continue   # reading every source file is cheap, parsing its imports is not
        for module in imported_modules(path, src):
            parts = module.split(".")
            if top == "integrations" and parts[0] == "interfaces" and not path.stem.endswith("Tab"):
                found["interface_imports"].add((rel.as_posix(), module))
            if parts[0] != "integrations" or len(parts) < 2:
                continue
            if top in CORE_PACKAGES:
                found["core_by_name"].setdefault(rel.as_posix(), set()).add(parts[1])
                if any(part.endswith("Tab") for part in parts[2:]):
                    found["core_tab_imports"].add((rel.as_posix(), module))
            elif len(rel.parts) > 2 and rel.parts[1] != parts[1]:
                found["folder_to_folder"].add((rel.parts[1], parts[1]))
    return found


@pytest.fixture(scope="module")
def tree():
    return survey(SRC_DIR)


def test_only_a_tab_module_of_a_folder_imports_an_interface(tree):
    assert tree["interface_imports"] == set()


def test_core_never_imports_a_tab_module(tree):
    assert tree["core_tab_imports"] == set()


def test_core_imports_a_folder_by_name_only_where_listed(tree):
    assert tree["core_by_name"] == CORE_IMPORTS_BY_NAME


def test_the_launcher_imports_no_folder_by_name():
    launcher = AISERVER_DIR / "floxy_ai_server.py"
    folders = {module.split(".")[1] for module in imported_modules(launcher, launcher.parent)
               if module.startswith("integrations.")}
    assert folders == set()


def test_a_folder_imports_another_folder_only_where_listed(tree):
    assert tree["folder_to_folder"] == FOLDER_IMPORTS_FOLDER


def test_every_folder_is_a_package_the_discovery_walks():
    folders = {p.name for p in (SRC_DIR / "integrations").iterdir() if p.is_dir() and p.name != "__pycache__"}
    assert folders and all((SRC_DIR / "integrations" / name / "__init__.py").exists() for name in folders)


def write(root: Path, name: str, source: str):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_the_survey_sees_each_kind_of_breach(tmp_path):
    write(tmp_path, "integrations/lamp/LampIntegration.py",
          "from interfaces.console.tabs.TabView import TabView\nfrom ..kettle.KettleIntegration import Kettle\n")
    write(tmp_path, "integrations/lamp/LampTab.py", "from interfaces.console.tabs.TabView import TabView\n")
    write(tmp_path, "assistant/Brain.py",
          '"""Thinks.\n\nfrom the lamp the kettle is lit, import nothing of it\n"""\n'
          "def think():\n    from integrations.lamp import (\n        LampTab,\n    )\n"
          "    import integrations.kettle.KettleIntegration\n")
    write(tmp_path, "interfaces/console/Lamps.py", "from integrations.lamp.LampTab import LampTab\n")

    found = survey(tmp_path)

    assert {path for path, _ in found["interface_imports"]} == {"integrations/lamp/LampIntegration.py"}
    assert found["core_tab_imports"] == {("assistant/Brain.py", "integrations.lamp.LampTab")}
    assert found["core_by_name"] == {"assistant/Brain.py": {"lamp", "kettle"}}
    assert found["folder_to_folder"] == {("lamp", "kettle")}
