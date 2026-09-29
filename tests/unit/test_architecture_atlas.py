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


def test_descriptions_and_tour_match_the_source_tree(atlas):
    assert atlas.problems(atlas.build_tree()) == []


def test_a_renamed_tour_module_is_reported(atlas, monkeypatch):
    tree = atlas.build_tree()
    monkeypatch.setattr(atlas, "TOUR", [("src/assistant/Renamed.py", "Gone.")])
    assert any("TOUR step 1" in p for p in atlas.problems(tree))


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
