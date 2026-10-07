"""The Keep Atlas (docs/atlas) stays true to the tree: CI publishes it from master and refuses to build when
descriptions.py names a module that is gone or a module has nothing to say."""
import importlib.util

import pytest

from common.Paths import REPO_DIR


@pytest.fixture(scope="module")
def atlas():
    spec = importlib.util.spec_from_file_location("atlas_build", REPO_DIR / "docs" / "atlas" / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE_NODES_KEY = "floxy/atlas-modules"


@pytest.fixture
def unchanged_modules_from_the_cache(atlas, request, monkeypatch):
    """build_tree parses every source file, 0.2 s of compile() on each run. A module's node depends
    only on its file, build.py and descriptions.py, so the nodes are kept in the pytest cache
    and only the files changed since the last run are parsed again."""
    cache = request.config.cache
    stamp = [(REPO_DIR / "docs" / "atlas" / name).stat().st_mtime_ns for name in ("build.py", "descriptions.py")]
    stored = cache.get(MODULE_NODES_KEY, None) or {}
    known = stored.get("modules", {}) if stored.get("stamp") == stamp else {}
    seen = {}
    parse = atlas._module

    def module(path):
        stat = path.stat()
        version = [stat.st_mtime_ns, stat.st_size]
        hit = known.get(str(path))
        node = hit["node"] if hit and hit["version"] == version else parse(path)
        seen[str(path)] = {"version": version, "node": node}
        return node

    monkeypatch.setattr(atlas, "_module", module)
    yield
    if seen != known:
        cache.set(MODULE_NODES_KEY, {"stamp": stamp, "modules": seen})


def test_descriptions_and_tour_match_the_source_tree(atlas, unchanged_modules_from_the_cache):
    assert atlas.problems(atlas.build_tree()) == []


def test_a_renamed_tour_module_is_reported(atlas, monkeypatch):
    tree = {"id": "root", "kind": "system", "desc": "A tree without the module the tour names."}
    monkeypatch.setattr(atlas, "TOUR", [("src/assistant/Renamed.py", "Gone.")])
    assert any("TOUR step 1" in p for p in atlas.problems(tree))


def test_a_top_level_group_in_a_colour_the_template_lacks_is_reported(atlas):
    group = {"id": "src/newcomer", "kind": "package", "desc": "A group.", "hue": "chartreuse"}
    tree = {"id": "root", "kind": "system", "desc": "A tree.", "children": [group]}
    assert any("src/newcomer" in p and "chartreuse" in p for p in atlas.problems(tree))


def test_first_sentence_stops_at_the_sentence_not_an_abbreviation(atlas):
    doc = "Dims the lights, e.g. in the kitchen.\n\nThe rest explains how."
    assert atlas.first_sentence(doc) == "Dims the lights, e.g. in the kitchen."


def test_first_sentence_drops_plan_references(atlas):
    assert atlas.first_sentence("Reads the feed (plans/FEED_PLAN.md §3) from a cursor.") == "Reads the feed from a cursor."


@pytest.fixture
def module_in(atlas, tmp_path, monkeypatch):
    """Writes a source file under a stand-in repository root and maps it."""
    monkeypatch.setattr(atlas, "REPO", tmp_path)

    def write(name, source):
        path = tmp_path / "src" / "pkg" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        return atlas._module(path)
    return write


def test_the_main_class_docstring_describes_the_module_over_the_module_docstring(module_in):
    node = module_in("Widget.py", '''"""Module-level blurb that is not the description."""


class Widget:
    """Draws one widget. Then more detail."""
''')
    assert node["desc"] == "Draws one widget."
    assert "children" not in node, "a single class is the module, not a child of it"


def test_exception_and_private_classes_are_not_mapped(module_in):
    node = module_in("store.py", '''"""Keeps things."""


class StoreError(Exception):
    """Raised when the store is full."""


class _Cursor:
    """Walks the store."""


class Store:
    """Holds the things."""


class Shelf:
    """One shelf of the store."""
''')
    assert node["desc"] == "Keeps things."
    assert [c["name"] for c in node["children"]] == ["Store", "Shelf"]


def test_more_info_is_the_docstring_after_its_first_sentence(module_in):
    node = module_in("Widget.py", '''class Widget:
    """Draws one widget. It is redrawn
    on every frame.

    Sizes:
      - small
    """
''')
    assert node["desc"] == "Draws one widget."
    assert node["more"] == "It is redrawn\non every frame.\n\nSizes:\n  - small"


def test_a_one_sentence_docstring_has_no_more_info(module_in):
    assert module_in("lamp.py", '"""Dims the lamp, e.g. at night."""\n')["more"] == ""


def test_more_info_keeps_the_whole_docstring_when_the_description_is_an_override(atlas, module_in, monkeypatch):
    monkeypatch.setitem(atlas.MODULES, "src/pkg/lamp.py", "Runs the lamp.")
    node = module_in("lamp.py", '"""Dims the lamp. Slowly."""\n')
    assert (node["desc"], node["more"]) == ("Runs the lamp.", "Dims the lamp. Slowly.")


# --- the toolkits the atlas lists ------------------------------------------------------------

def test_the_atlas_reads_a_folder_toolkits_front_matter_the_way_the_server_does(atlas, tmp_path):
    """build.py runs in CI with the stdlib alone, so it reads toolkit.md front matter with a
    reader of its own, which must agree with the YAML loader the server uses — on every
    toolkit of the repository and on a summary that is quoted because it holds a colon."""
    from assistant.commands import Toolkits

    for toolkit in Toolkits.discover():
        meta = atlas._front_matter(toolkit.file)
        assert (meta["name"], meta["summary"], tuple(meta["commands"])) == \
               (toolkit.name, toolkit.summary, toolkit.commands), toolkit.file

    path = tmp_path / "lamp" / Toolkits.FOLDER_TOOLKIT_FILE
    path.parent.mkdir()
    path.write_text('---\nname: lamp\nsummary: "the lamps: on, off"\ncommands: [lamp_on, "lamp_off"]\n---\nbody\n',
                    encoding="utf-8")
    toolkit = Toolkits.read_folder_toolkit(path)
    meta = atlas._front_matter(path)
    assert (meta["name"], meta["summary"], tuple(meta["commands"])) == (toolkit.name, toolkit.summary, toolkit.commands)


def test_the_atlas_lists_every_toolkit_under_the_file_that_holds_it(atlas, unchanged_modules_from_the_cache):
    """A core toolkit is in res/toolkits, one an integration owns is in its folder: the atlas
    node of each points at the file the server reads, with the commands the server registered."""
    from assistant.commands.Toolkits import TOOLKITS

    res = next(g for g in atlas.build_tree()["children"] if g["id"] == "res")
    group = next(c for c in res["children"] if c["id"] == "res/toolkits")
    listed = {node["path"]: node for node in group["children"]}

    expected = {toolkit.path.relative_to(REPO_DIR).as_posix(): toolkit for toolkit in TOOLKITS}
    assert set(listed) == set(expected)
    for path, toolkit in expected.items():
        assert listed[path]["commands"] == list(toolkit.commands), path
        assert listed[path]["lines"] == len(toolkit.path.read_text(encoding="utf-8").splitlines()), path
