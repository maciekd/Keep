import json
from unittest.mock import Mock

import pytest

from integrations.podcasts import generate_report as report
from integrations.podcasts.ReportLLM import Completion, ReportLLMError
from .conftest import write_bundle


class ScriptedLLM:
    """Answers each call from `answers` in turn (an exception is raised), recording the user messages."""
    label = "openai:gpt-test"

    def __init__(self, *answers):
        self.answers = list(answers)
        self.users = []

    def complete(self, system, user, **kwargs):
        self.users.append(user)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return Completion(text=answer, seconds=0.1)


EXPLAINER_1 = "| Term | Explanation |\n|---|---|\n| **Tokamak** | A magnetic plasma bottle. |"


def test_the_artifacts_meta_and_manifest_land_in_the_episode_folder(tmp_path):
    bundle = write_bundle(tmp_path / "bundles")
    out = tmp_path / "Moonshots" / "moonshots-412"
    llm = ScriptedLLM(EXPLAINER_1, "Nothing worth unpacking here.", "## 1. Idea\n- **Hook:** x", "- one\n- two")

    manifest = report.generate(bundle, out, llm, ["explainer", "linkedin_ideas", "tldr"], "Moonshots")

    assert sorted(p.name for p in out.iterdir()) == ["explainer.md", "linkedin_ideas.md", "manifest.json",
                                                     "meta.json", "tldr.md"]
    assert json.loads((out / "manifest.json").read_text()) == manifest
    assert manifest["episode_number"] == 412 and set(manifest["artifacts"]) == {"explainer", "linkedin_ideas", "tldr"}
    meta = json.loads((out / "meta.json").read_text())
    assert meta["video_id"] == "vid412" and meta["model_used"] == "openai:gpt-test"
    assert {"download_s", "transcribe_s", "explainer_s", "linkedin_ideas_s", "tldr_s"} <= set(meta["timings"])


def test_the_manifest_is_written_after_every_artifact(tmp_path, monkeypatch):
    bundle = write_bundle(tmp_path / "bundles")
    out = tmp_path / "out"
    order = []
    real = report.write_atomically
    monkeypatch.setattr(report, "write_atomically", lambda path, text: (order.append(path.name), real(path, text)))

    report.generate(bundle, out, ScriptedLLM("a", "b", "c"), ["explainer", "tldr"], "Moonshots")

    assert order[-1] == "manifest.json"
    assert not list(out.glob("*.tmp"))


def test_the_explainer_has_a_wikilink_toc_and_one_section_per_chapter(tmp_path):
    bundle = write_bundle(tmp_path / "bundles")
    text = report.explainer(ScriptedLLM(EXPLAINER_1, "Plain."), report.load_meta(bundle), report.load_chapters(bundle))

    assert text.startswith("---\nepisode: ")
    assert "- [[#Opening — 0:00 - 59:00|Opening (0:00 - 59:00)]]" in text
    assert "## Fusion — 2:05 - 59:00" in text


def test_a_term_one_chapter_defines_is_handed_to_the_next(tmp_path):
    bundle = write_bundle(tmp_path / "bundles")
    llm = ScriptedLLM(EXPLAINER_1, "Plain.")
    report.explainer(llm, report.load_meta(bundle), report.load_chapters(bundle))
    assert "Already covered" not in llm.users[0]
    assert "Tokamak" in llm.users[1]


def test_a_failed_segment_leaves_a_callout_and_the_rest_goes_on(tmp_path):
    bundle = write_bundle(tmp_path / "bundles")
    text = report.explainer(ScriptedLLM(ReportLLMError("timeout"), "Plain."), report.load_meta(bundle),
                            report.load_chapters(bundle))
    assert "> [!error]" in text and "Plain." in text


def test_every_segment_failing_is_no_explainer(tmp_path):
    bundle = write_bundle(tmp_path / "bundles")
    with pytest.raises(report.ReportError):
        report.explainer(ScriptedLLM(ReportLLMError("a"), ReportLLMError("b")), report.load_meta(bundle),
                         report.load_chapters(bundle))


def test_a_regeneration_keeps_the_other_artifacts_and_says_what_was_redone(tmp_path):
    bundle = write_bundle(tmp_path / "bundles")
    out = tmp_path / "out"
    report.generate(bundle, out, ScriptedLLM("x", "y", "ideas", "tldr"), ["explainer", "linkedin_ideas", "tldr"], "M")
    manifest = report.generate(bundle, out, ScriptedLLM("new tldr"), ["tldr"], "M", regenerate=True)
    assert manifest["regenerated"] == ["tldr"] and set(manifest["artifacts"]) == {"explainer", "linkedin_ideas", "tldr"}
    assert "new tldr" in (out / "tldr.md").read_text()


def test_a_short_separator_row_is_widened_to_the_header():
    fixed = report.fix_table_separators("| A | B |\n|---|\n| 1 | 2 |")
    assert fixed.split("\n")[1] == "|---|---|"


def test_leaked_narration_before_the_first_table_is_dropped():
    text = "I have everything I need, both verified.\n\n| Term | Explanation |"
    assert report.strip_leaked_process_narration(text) == "| Term | Explanation |"
    assert report.strip_leaked_process_narration("A real intro.\n\n| T |") == "A real intro.\n\n| T |"


def test_the_ideas_are_split_by_number():
    ideas = report.ideas_of("# Ideas\n\n## 1. First\n- a\n\n## 2. Second\n- **Source:** [1:02:05]\n")
    assert sorted(ideas) == [1, 2] and ideas[2].startswith("## 2. Second") and "First" not in ideas[2]


def test_the_draft_excerpt_comes_from_the_chapter_of_the_source_timestamp(tmp_path):
    bundle = write_bundle(tmp_path / "bundles")
    excerpt = report.excerpt_around(report.load_chapters(bundle), "- **Source:** [2:30] Fusion")
    assert excerpt.startswith("## Fusion") and "tokamak" in excerpt
    assert report.excerpt_around(report.load_chapters(bundle), "no timestamp") == ""
