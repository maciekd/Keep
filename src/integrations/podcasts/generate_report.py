"""Writes the reports of one episode bundle — explainer, LinkedIn ideas, TL;DR — into the show's output folder.

Reads only what fetch_and_transcribe.py writes (video_meta.json, chapters/), so the source can be
swapped without touching this module. The explainer is the concept-and-reference companion of
prompts/explainer.md, one LLM call per chapter with the web_search tool, the terms already
covered handed to the next chapter so nothing is defined twice; the LinkedIn ideas and the TL;DR
are one call each over the whole transcript. The files land in `<output_folder>/<slug>/`:
explainer.md, linkedin_ideas.md, tldr.md and meta.json, then manifest.json last, written to a
.tmp and renamed — its presence is the "done" signal the integration's feed waits for.
"""
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from common.logging import error, info, note, trace
from integrations.podcasts.ReportLLM import ReportLLM, ReportLLMError

PROMPTS_DIR = Path(__file__).with_name("prompts")
MANIFEST = "manifest.json"
META = "meta.json"
ARTIFACT_FILES = {"explainer": "explainer.md", "linkedin_ideas": "linkedin_ideas.md", "tldr": "tldr.md"}

TABLE_TERM_RE = re.compile(r'^\|\s*\*\*(.+?)\*\*\s*\|', re.MULTILINE)
HEADING_RE = re.compile(r'^###\s+(.+)$', re.MULTILINE)
TABLE_SEP_RE = re.compile(r'^\|[\s:|-]+\|\s*$')
LEAKED_NARRATION_RE = re.compile(
    r"\bI(?:'ve| have|'m| am) .{0,40}\b(verified|what I need|everything I need|ready)\b|"
    r"\blet me check\b|\bready to (?:write|answer)\b", re.IGNORECASE)
EPISODE_NUMBER_RE = re.compile(r"-(\d{2,5})$")


class ReportError(Exception):
    """An artifact could not be generated."""


@dataclass
class Chapter:
    id: str
    title: str
    range: str
    body: str

    @property
    def heading(self) -> str:
        return f"{self.title} — {self.range}"


def prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def load_meta(bundle: Path) -> dict:
    return json.loads((Path(bundle) / "video_meta.json").read_text(encoding="utf-8"))


def load_chapters(bundle: Path) -> list[Chapter]:
    chapters = []
    for path in sorted((Path(bundle) / "chapters").glob("*.txt")):
        lines = path.read_text(encoding="utf-8").split("\n")
        chapters.append(Chapter(path.name.split("_")[0], lines[0].removeprefix("CHAPTER: ").strip(),
                                lines[1].removeprefix("TIME: ").strip(), "\n".join(lines[3:]).strip()))
    return chapters


def episode_number(slug: str) -> Optional[int]:
    match = EPISODE_NUMBER_RE.search(slug)
    return int(match.group(1)) if match else None


def whole_transcript(chapters: list[Chapter]) -> str:
    return "\n\n".join(f"## {c.heading}\n\n{c.body}" for c in chapters)


def episode_header(meta: dict) -> str:
    return (f'Episode: "{meta["title"]}"\n'
            f'Source: {meta.get("uploader") or "unknown"}, uploaded {meta.get("upload_date") or "unknown"}, '
            f'{meta["video_url"]}\n')


# --- the explainer's repairs, ported unchanged from the Hermes pipeline ---------------------

def fix_table_separators(markdown: str) -> str:
    """Repairs a table separator row with fewer columns than its header, which Obsidian cannot render."""
    lines, out, i = markdown.split("\n"), [], 0
    while i < len(lines):
        line = lines[i].strip()
        if i + 1 < len(lines) and line.startswith("|") and line.endswith("|") and TABLE_SEP_RE.match(lines[i + 1].strip()):
            columns = len(line.strip("|").split("|"))
            if len(lines[i + 1].strip().strip("|").split("|")) != columns:
                out += [lines[i], "|" + "|".join(["---"] * columns) + "|"]
                i += 2
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def strip_leaked_process_narration(markdown: str) -> str:
    """Drops a leaked "I have what I need…" line in front of a segment's first table, heading or quote."""
    lines = markdown.split("\n")
    first = next((i for i, line in enumerate(lines) if line.strip()[:1] in ("|", ">") or line.strip().startswith("###")), None)
    if not first:
        return markdown
    leading = "\n".join(lines[:first]).strip()
    return "\n".join(lines[first:]) if leading and LEAKED_NARRATION_RE.search(leading) else markdown


def covered_terms(markdown: str) -> list[str]:
    seen, ordered = set(), []
    for term in TABLE_TERM_RE.findall(markdown) + HEADING_RE.findall(markdown):
        term = term.strip()
        if term and term.lower() not in seen:
            seen.add(term.lower())
            ordered.append(term)
    return ordered


def chapter_message(meta: dict, chapter: Chapter, index: int, total: int, covered: list[str]) -> str:
    covered_block = ""
    if covered:
        covered_block = ("Already covered in earlier segments of this same episode — do NOT redefine these from "
                         "scratch. If one reappears here with nothing new to add, just use it naturally in prose. "
                         "Only add a table row or section for one of these if this segment reveals something "
                         "genuinely new about it (a correction, a materially different context, real added detail) "
                         "— and if so, write only the new part:\n" + ", ".join(covered) + "\n\n")
    return (f'{episode_header(meta)}'
            f'This is segment {index + 1} of {total}: "{chapter.title}" ({chapter.range})\n\n'
            f'{covered_block}'
            f'--- transcript segment (auto-generated, may contain transcription errors) ---\n'
            f'{chapter.body}\n'
            f'--- end transcript segment ---\n\n'
            f'Produce the concept/reference notes for this segment only, following the system instructions exactly.')


def frontmatter(meta: dict, model: str, kind: str) -> str:
    return (f'---\n'
            f'episode: {json.dumps(meta["title"], ensure_ascii=False)}\n'
            f'kind: {kind}\n'
            f'model: "{model}"\n'
            f'source: "{meta["video_url"]}"\n'
            f'generated: "{time.strftime("%Y-%m-%d %H:%M")}"\n'
            f'---\n\n')


# --- the artifacts ----------------------------------------------------------------------------

def explainer(llm: ReportLLM, meta: dict, chapters: list[Chapter],
              progress: Callable[[str], None] = lambda step: None) -> str:
    system = prompt("explainer")
    covered: list[str] = []
    sections, failures = [], 0
    for index, chapter in enumerate(chapters):
        progress(f"explainer segment {index + 1}/{len(chapters)}")
        trace(f"Podcasts: explainer segment {index + 1}/{len(chapters)} {chapter.title!r}")
        try:
            answer = llm.complete(system, chapter_message(meta, chapter, index, len(chapters), covered),
                                  with_search=True).text
        except ReportLLMError as e:
            failures += 1
            note(f"Podcasts: explainer segment {chapter.title!r} failed: {e}")
            sections.append(f"## {chapter.heading}\n\n> [!error] Generation failed for this segment\n> {e}\n")
            continue
        sections.append(f"## {chapter.heading}\n\n{fix_table_separators(strip_leaked_process_narration(answer)).strip()}\n")
        known = {t.lower() for t in covered}
        covered += [t for t in covered_terms(answer) if t.lower() not in known]
    if failures == len(chapters):
        raise ReportError(f"every one of the {len(chapters)} explainer segments failed")
    toc = "\n".join(f"- [[#{c.heading}|{c.title} ({c.range})]]" for c in chapters)
    return (f"{frontmatter(meta, llm.label, 'explainer')}# {meta['title']}\n\n{toc}\n\n---\n\n"
            + "\n---\n\n".join(sections))


def linkedin_ideas(llm: ReportLLM, meta: dict, chapters: list[Chapter]) -> str:
    answer = llm.complete(prompt("linkedin_ideas"), f"{episode_header(meta)}\n{whole_transcript(chapters)}").text
    return f"{frontmatter(meta, llm.label, 'linkedin_ideas')}# LinkedIn ideas — {meta['title']}\n\n{answer}\n"


def tldr(llm: ReportLLM, meta: dict, chapters: list[Chapter]) -> str:
    answer = llm.complete(prompt("tldr"), f"{episode_header(meta)}\n{whole_transcript(chapters)}", max_tokens=2000).text
    return f"{frontmatter(meta, llm.label, 'tldr')}# TL;DR — {meta['title']}\n\n{answer}\n"


GENERATORS = {"explainer": explainer, "linkedin_ideas": linkedin_ideas, "tldr": tldr}

IDEA_RE = re.compile(r"^##\s+(\d+)\.", re.MULTILINE)
TIMESTAMP_RE = re.compile(r"\[(\d{1,2}(?::\d{2}){1,2})\]")
EXCERPT_CHARS = 8000


def ideas_of(text: str) -> dict[int, str]:
    """The numbered ideas of a linkedin_ideas.md, by number."""
    marks = list(IDEA_RE.finditer(text))
    return {int(m.group(1)): text[m.start():marks[i + 1].start() if i + 1 < len(marks) else len(text)].strip()
            for i, m in enumerate(marks)}


def seconds_of(stamp: str) -> int:
    seconds = 0
    for part in stamp.split(":"):
        seconds = seconds * 60 + int(part)
    return seconds


def excerpt_around(chapters: list[Chapter], idea: str) -> str:
    """The transcript of the chapter the idea's first timestamp falls in, cut to about
    EXCERPT_CHARS around that moment; "" when the idea carries no timestamp."""
    match = TIMESTAMP_RE.search(idea)
    if not match or not chapters:
        return ""
    at = seconds_of(match.group(1))
    chapter = chapters[0]
    for candidate in chapters:
        if seconds_of(candidate.range.split(" - ")[0]) <= at:
            chapter = candidate
    position = chapter.body.find(f"[{match.group(1)}]")
    start = max(0, (position if position >= 0 else 0) - EXCERPT_CHARS // 2)
    return f"## {chapter.heading}\n\n{chapter.body[start:start + EXCERPT_CHARS]}"


def draft_post(llm: ReportLLM, meta: dict, idea: str, chapters: list[Chapter]) -> str:
    excerpt = excerpt_around(chapters, idea)
    user = (f"{episode_header(meta)}\n--- the idea ---\n{idea}\n--- end idea ---\n\n"
            + (f"--- transcript around the source ---\n{excerpt}\n--- end transcript ---\n" if excerpt else ""))
    return llm.complete(prompt("draft"), user, max_tokens=2000, temperature=0.6).text


def write_atomically(path: Path, text: str):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def generate(bundle: Path, output_dir: Path, llm: ReportLLM, kinds: list[str], show: str,
             progress: Callable[[str], None] = lambda step: None, regenerate: bool = False) -> dict:
    """Generates `kinds` from the bundle into `output_dir`, then writes meta.json and the manifest;
    returns the manifest. `regenerate` keeps the episode's other artifacts and records which were redone."""
    meta = load_meta(bundle)
    chapters = load_chapters(bundle)
    if not chapters:
        raise ReportError(f"no chapters in {bundle}/chapters")
    output_dir.mkdir(parents=True, exist_ok=True)
    old_meta = _read_json(output_dir / META) if regenerate else {}
    timings = dict(old_meta.get("timings") or meta.get("timings") or {})
    info(f"Podcasts: generating {', '.join(kinds)} for {meta['slug']} ({len(chapters)} chapters) with {llm.label}")
    for kind in kinds:
        progress(kind)
        started = time.time()
        if kind == "explainer":
            text = explainer(llm, meta, chapters, progress)
        else:
            try:
                text = GENERATORS[kind](llm, meta, chapters)
            except ReportLLMError as e:
                raise ReportError(f"{kind}: {e}") from e
        write_atomically(output_dir / ARTIFACT_FILES[kind], text)
        timings[f"{kind}_s"] = round(time.time() - started, 1)
        info(f"Podcasts: {meta['slug']} {kind} written in {timings[f'{kind}_s']:.0f}s ({len(text)} chars)")

    artifacts = {kind: name for kind, name in ARTIFACT_FILES.items() if (output_dir / name).exists()}
    now = time.time()
    episode_meta = {
        "slug": meta["slug"], "video_id": meta["video_id"], "title": meta["title"], "url": meta["video_url"],
        "show": show, "upload_date": meta.get("upload_date"),
        "generated_at": old_meta.get("generated_at") or _iso(now),
        "transcript_source": meta.get("transcript_source"), "model_used": llm.label,
        "duration": meta.get("duration_s"), "chapters": len(chapters), "timings": timings,
    }
    if regenerate:
        episode_meta["regenerated_at"] = _iso(now)
    write_atomically(output_dir / META, json.dumps(episode_meta, indent=2, ensure_ascii=False))
    manifest = {
        "slug": meta["slug"], "video_id": meta["video_id"], "show": show, "title": meta["title"],
        "episode_number": episode_number(meta["slug"]), "url": meta["video_url"],
        "finished_at": now, "artifacts": artifacts, "model_used": llm.label,
    }
    if regenerate:
        manifest["regenerated"] = kinds
    write_atomically(output_dir / MANIFEST, json.dumps(manifest, indent=2, ensure_ascii=False))
    info(f"Podcasts: {meta['slug']} done — manifest written to {output_dir}")
    return manifest


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(ts))


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        error(f"Podcasts: {path} could not be read: {e}")
        return {}
