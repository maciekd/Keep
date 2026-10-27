import json

from integrations.podcasts.PodcastLibrary import PodcastLibrary

from .conftest import write_bundle


def finished(show_dir, slug, at, title="Robots Learn To Juggle", **artifacts):
    folder = show_dir / slug
    folder.mkdir()
    for name, text in artifacts.items():
        (folder / f"{name}.md").write_text(text)
    (folder / "manifest.json").write_text(json.dumps({"slug": slug, "title": title, "finished_at": at}))


def library(config):
    return PodcastLibrary(config.shows(), config.state_dir() / "episodes")


def test_the_episodes_are_newest_first_with_the_older_pipelines_files_among_them(config, show_dir):
    (show_dir / "moonshots-300.md").write_text('---\nepisode: "Mars By Tuesday | EP #300"\n---\n')
    finished(show_dir, "moonshots-412", 2e9, explainer="x", tldr="y")
    (show_dir / "notes.md").write_text("not an episode")

    episodes = library(config).episodes()

    assert [e.slug for e in episodes] == ["moonshots-412", "moonshots-300"]
    assert set(episodes[0].artifacts) == {"explainer", "tldr"} and not episodes[0].legacy
    assert episodes[1].legacy and episodes[1].title == "Mars By Tuesday | EP #300"
    assert episodes[0].label() == "Moonshots #412: Robots Learn To Juggle"


def test_a_broken_manifest_is_skipped(config, show_dir):
    (show_dir / "moonshots-5").mkdir()
    (show_dir / "moonshots-5" / "manifest.json").write_text("{not json")
    assert library(config).episodes() == []


def test_search_finds_the_reports_and_the_kept_transcripts(config, show_dir):
    finished(show_dir, "moonshots-412", 2e9, explainer="| **Tokamak** | A plasma bottle at Alderbank. |")
    write_bundle(config.state_dir() / "episodes")

    hits = library(config).search("alderbank")

    assert any(h.startswith("[moonshots-412 explainer]") for h in hits)
    assert any(h.startswith("[moonshots-412 transcript (01_2-05)]") for h in hits)
    assert library(config).search("alderbank zeppelin") == []
