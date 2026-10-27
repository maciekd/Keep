import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from assistant.integration.Integration import Integration
from common.DevelopmentConfig import RUNNING_UNDER_TEST
from common.logging import error, info, trace
from integrations.podcasts.PodcastLibrary import Episode, PodcastLibrary
from integrations.podcasts.PodcastsConfig import SHOWS_KEY, PodcastsConfig
from integrations.podcasts.PodcastsPipeline import PodcastsPipeline

TICK_SECONDS = 5 * 60


class PodcastsIntegration(Integration):
    """Podcast episode reports: explainer, LinkedIn ideas and TL;DR.

    Every morning after the check time, fetches each new episode of the listed YouTube playlists,
    transcribes it and writes an explainer, LinkedIn ideas and a TL;DR into the show's folder,
    with a message when it starts and when the report is ready.

    A timer asks the pipeline every five minutes whether a check is due (past podcasts.check_time,
    an hour since the last one) — the autonomous check has no hook for an integration. The work is
    PodcastsPipeline's, on threads of its own: fetch_and_transcribe.py (yt-dlp, mlx-whisper; the
    swappable source) writes the episode bundle, generate_report.py the reports, through the
    OpenAI API by default or a local omlx server, and manifest.json last, the "done" signal. The
    commands read the output folders through PodcastLibrary."""

    def __init__(self, config: Optional[PodcastsConfig] = None, pipeline: Optional[PodcastsPipeline] = None):
        super().__init__()
        self.config = config or PodcastsConfig()
        self.pipeline = pipeline or PodcastsPipeline(self.config)
        self._stopped = threading.Event()
        self._timer: Optional[threading.Timer] = None
        if RUNNING_UNDER_TEST:
            trace("Podcasts: running under test, the check timer is not started")
        else:
            self._schedule(TICK_SECONDS)

    def _schedule(self, delay: float):
        if self._stopped.is_set():
            return
        self._timer = threading.Timer(delay, self._tick)
        self._timer.daemon = True
        self._timer.start()

    def _tick(self):
        if self._stopped.is_set():
            trace("Podcasts: tick after the integration was dropped, skipped")
            return
        try:
            if self.config.shows():
                self.pipeline.maybe_start()
        except Exception as e:
            error(f"Podcasts: the check could not be started: {type(e).__name__}: {e}")
        finally:
            self._schedule(TICK_SECONDS)

    def _on_reset(self):
        self._stopped.set()
        if self._timer is not None:
            self._timer.cancel()
        info("Podcasts dropped: the check timer is stopped")

    @property
    def name(self) -> str:
        return "Podcasts"

    @property
    def config_keys(self) -> List[str]:
        return [SHOWS_KEY]

    def library(self) -> PodcastLibrary:
        return self.pipeline.library()

    def check_health(self) -> bool:
        shows = self.config.shows()
        missing = [str(s.output_folder) for s in shows if not s.output_folder.is_dir()]
        if missing:
            raise RuntimeError(f"output folder missing: {', '.join(missing)}")
        if self.pipeline.last_problem:
            raise RuntimeError(self.pipeline.last_problem)
        return bool(shows)

    # --- commands ----------------------------------------------------------------------------

    def list_episodes(self, limit: int = 5) -> str:
        episodes = self.library().episodes()[:max(1, limit)]
        if not episodes:
            return "No podcast reports yet."
        status = self.pipeline.status()
        lines = [self._episode_line(e) for e in episodes]
        if self.pipeline.running and status.get("slug"):
            lines.insert(0, f"In progress: {status['slug']} — {status.get('step')}")
        return "\n".join(lines)

    def _episode_line(self, episode: Episode) -> str:
        when = datetime.fromtimestamp(episode.finished_at).strftime("%Y-%m-%d")
        kinds = ", ".join(episode.artifacts) or "none"
        extra = " (older pipeline: explainer only, no transcript)" if episode.legacy else ""
        return f"- {episode.slug} — {episode.label()} — {when} — artifacts: {kinds}{extra}"

    def load_artifact(self, slug: str, kind: str) -> str:
        episode = self.library().episode(slug)
        if episode is None:
            return f"load_podcast_artifact failed: no episode {slug!r} — podcast_episodes lists them"
        path: Optional[Path] = episode.artifacts.get(kind.strip())
        if path is None:
            return f"load_podcast_artifact failed: {episode.slug} has {', '.join(episode.artifacts) or 'no artifacts'}, not {kind!r}"
        info(f"Podcasts: loading the {kind} of {episode.slug} ({path})")
        return path.read_text(encoding="utf-8")

    def search(self, query: str) -> str:
        hits = self.library().search(query)
        info(f"Podcasts: search {query!r} — {len(hits)} hit(s)")
        return "\n".join(hits) if hits else f"Nothing in the podcast reports or transcripts matches {query!r}."
