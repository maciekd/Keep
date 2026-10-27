"""The finished podcast reports in the shows' output folders, and the transcripts the pipeline kept.

An episode is a `<output_folder>/<slug>/` with a manifest.json (this integration's), or a flat
`<output_folder>/<slug>.md` the Hermes pipeline wrote before it — an explainer alone, read but
never touched. The transcripts are the bundles under the state dir's `episodes/<slug>/`.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from common.logging import error, info, trace
from integrations.podcasts.PodcastsConfig import Show
from integrations.podcasts.generate_report import ARTIFACT_FILES, MANIFEST, META, episode_number

TITLE_RE = re.compile(r'^episode:\s*"?(.*?)"?\s*$', re.MULTILINE)
SEARCH_CONTEXT = 1      # lines either side of a hit
MAX_HITS_PER_FILE = 3
MAX_HITS = 30


@dataclass
class Episode:
    show: str
    slug: str
    title: str
    finished_at: float
    artifacts: dict[str, Path] = field(default_factory=dict)
    folder: Optional[Path] = None       # None: a flat file of the Hermes pipeline
    manifest: dict = field(default_factory=dict)

    @property
    def number(self) -> Optional[int]:
        return episode_number(self.slug)

    @property
    def legacy(self) -> bool:
        return self.folder is None

    def label(self) -> str:
        """"Moonshots #305: <title>" (the slug when the title has no number)."""
        return f"{self.show} #{self.number}: {self.title}" if self.number else f"{self.show} {self.slug}: {self.title}"


def read_manifest(path: Path) -> Optional[dict]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        float(manifest["finished_at"])
        return manifest
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError) as e:
        error(f"Podcasts: manifest {path} is unreadable, skipped: {type(e).__name__}: {e}")
        return None


class PodcastLibrary:
    """Reads the output folders of `shows` and the bundles under `bundles_root`."""

    def __init__(self, shows: list[Show], bundles_root: Path):
        self.shows = shows
        self.bundles_root = bundles_root

    def manifests(self) -> list[tuple[Show, dict]]:
        found = []
        for show in self.shows:
            for path in sorted(show.output_folder.glob(f"*/{MANIFEST}")):
                if (manifest := read_manifest(path)) is not None:
                    found.append((show, manifest))
        return sorted(found, key=lambda pair: float(pair[1]["finished_at"]))

    def episodes(self) -> list[Episode]:
        """Every episode of every show, newest first."""
        episodes: dict[str, Episode] = {}
        for show in self.shows:
            if not show.output_folder.is_dir():
                info(f"Podcasts: output folder {show.output_folder} of {show.name} does not exist")
                continue
            for path in show.output_folder.glob(f"{show.slug_prefix}-*.md"):
                episodes[path.stem] = Episode(show.name, path.stem, self._legacy_title(path) or path.stem,
                                              path.stat().st_mtime, {"explainer": path})
        for show, manifest in self.manifests():
            folder = show.output_folder / manifest["slug"]
            artifacts = {kind: folder / name for kind, name in ARTIFACT_FILES.items() if (folder / name).exists()}
            episodes[manifest["slug"]] = Episode(show.name, manifest["slug"], manifest.get("title") or manifest["slug"],
                                                 float(manifest["finished_at"]), artifacts, folder, manifest)
        return sorted(episodes.values(), key=lambda e: e.finished_at, reverse=True)

    def episode(self, slug: str) -> Optional[Episode]:
        wanted = slug.strip().lower()
        return next((e for e in self.episodes() if e.slug.lower() == wanted), None)

    def is_done(self, show: Show, slug: str) -> bool:
        """Whether the show has a report for `slug` already, from this pipeline or the one before it."""
        return (show.output_folder / slug / MANIFEST).exists() or (show.output_folder / f"{slug}.md").exists()

    def bundle(self, slug: str) -> Optional[Path]:
        path = self.bundles_root / slug
        return path if (path / "video_meta.json").exists() else None

    def meta(self, episode: Episode) -> dict:
        if episode.folder is None:
            return {}
        try:
            return json.loads((episode.folder / META).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _legacy_title(path: Path) -> Optional[str]:
        try:
            with path.open(encoding="utf-8") as f:
                head = f.read(2000)
        except OSError:
            return None
        match = TITLE_RE.search(head)
        return match.group(1).strip() if match else None

    def search(self, query: str) -> list[str]:
        """The lines matching every word of `query` (case-insensitive) in the reports and the
        transcripts, each with a line of context and where it is."""
        words = [w.lower() for w in query.split() if w.strip()]
        if not words:
            return []
        hits: list[str] = []
        for episode in self.episodes():
            sources = [(kind, path) for kind, path in episode.artifacts.items()]
            if (bundle := self.bundle(episode.slug)) is not None:
                sources += [("transcript", p) for p in sorted((bundle / "chapters").glob("*.txt"))]
            for kind, path in sources:
                hits += self._search_file(path, words, f"{episode.slug} {kind}")
                if len(hits) >= MAX_HITS:
                    trace(f"Podcasts: search {query!r} stopped at {MAX_HITS} hits")
                    return hits[:MAX_HITS]
        return hits

    @staticmethod
    def _search_file(path: Path, words: list[str], where: str) -> list[str]:
        try:
            lines = path.read_text(encoding="utf-8").split("\n")
        except OSError:
            return []
        hits = []
        for i, line in enumerate(lines):
            if all(w in line.lower() for w in words):
                context = " ".join(s.strip() for s in lines[max(0, i - SEARCH_CONTEXT):i + SEARCH_CONTEXT + 1] if s.strip())
                chapter = f" ({path.stem})" if where.endswith("transcript") else ""
                hits.append(f"[{where}{chapter}] {context[:600]}")
                if len(hits) >= MAX_HITS_PER_FILE:
                    break
        return hits
