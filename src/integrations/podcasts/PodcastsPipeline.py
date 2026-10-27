"""The podcast pipeline's driver: when a check is due, fetch the next episode of each show and write its reports.

A check is due once the day is past `podcasts.check_time` and an hour has gone by since the last
one (the integration's feed asks on every poll). One check takes at most one new episode per
show: fetch_and_transcribe.fetch_next, then generate_report.generate, the owner told when it
starts and when it is done. The state lives in the state dir (`podcasts.state_dir`, cache/podcasts/
by default): state/processed.json — the video ids done and the failed attempts per id, an
episode given up on after MAX_FAILED_ATTEMPTS —, state/status.json — the heartbeat of the run
in progress —, and episodes/<slug>/, the bundles, kept for podcast_search, podcast_draft and
podcast_regenerate. An episode the output folder already has a report for (the Hermes
pipeline's flat <slug>.md included) counts as processed.
"""
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from common.logging import error, info, note, trace
from integrations.podcasts import fetch_and_transcribe as source
from integrations.podcasts import generate_report as report
from integrations.podcasts.PodcastLibrary import PodcastLibrary
from integrations.podcasts.PodcastsConfig import PROVIDER_OPENAI, PodcastsConfig, Show
from integrations.podcasts.ReportLLM import ReportLLM, ReportLLMError

CHECK_EVERY_S = 60 * 60
MAX_FAILED_ATTEMPTS = 3


def describe_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}m{secs:02d}s" if minutes < 60 else f"{minutes // 60}h{minutes % 60:02d}m"


def describe_timings(timings: dict) -> str:
    names = {"download_s": "download", "transcribe_s": "transcribe", "explainer_s": "explainer",
             "linkedin_ideas_s": "LinkedIn ideas", "tldr_s": "TL;DR"}
    parts = [f"{label} {describe_seconds(timings[key])}" for key, label in names.items() if key in timings]
    return ", ".join(parts)


class PodcastsPipeline:
    """Runs the checks and the regenerations, one at a time, each on a thread of its own.

    The run in progress is the class's, not the instance's: a restarted integration builds a new
    pipeline while the old one's run goes on, and the two must not fetch the same episode."""
    _lock = threading.Lock()
    _thread: Optional[threading.Thread] = None

    def __init__(self, config: Optional[PodcastsConfig] = None, notify: Optional[Callable[[str], None]] = None,
                 llm_factory: Callable[[PodcastsConfig], ReportLLM] = lambda c: ReportLLM(c.llm())):
        self.config = config or PodcastsConfig()
        self._notify = notify or self._send_to_owner
        self._llm_factory = llm_factory
        self.last_check_at: Optional[float] = None
        self.last_problem: Optional[str] = None

    # --- paths and state ------------------------------------------------------------------

    @property
    def state_dir(self) -> Path:
        path = self.config.state_dir() / "state"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def bundles_root(self) -> Path:
        return self.config.state_dir() / "episodes"

    def library(self) -> PodcastLibrary:
        return PodcastLibrary(self.config.shows(), self.bundles_root)

    def load_state(self) -> dict:
        try:
            state = json.loads((self.state_dir / "processed.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            state = {}
        except (OSError, ValueError) as e:
            error(f"Podcasts: processed.json is unreadable, starting from an empty state: {e}")
            state = {}
        state.setdefault("processed", [])
        state.setdefault("failed_attempts", {})
        return state

    def _save_state(self, state: dict):
        report.write_atomically(self.state_dir / "processed.json", json.dumps(state, indent=2))

    def mark_processed(self, video_id: str):
        state = self.load_state()
        if video_id not in state["processed"]:
            state["processed"].append(video_id)
        state["failed_attempts"].pop(video_id, None)
        self._save_state(state)

    def record_failure(self, video_id: str) -> int:
        state = self.load_state()
        state["failed_attempts"][video_id] = state["failed_attempts"].get(video_id, 0) + 1
        self._save_state(state)
        return state["failed_attempts"][video_id]

    def heartbeat(self, **status):
        status.update(updated_at=time.time(), pid=os.getpid())
        try:
            report.write_atomically(self.state_dir / "status.json", json.dumps(status, indent=2, ensure_ascii=False))
        except OSError as e:
            error(f"Podcasts: status.json could not be written: {e}")

    def status(self) -> dict:
        try:
            return json.loads((self.state_dir / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    # --- scheduling -----------------------------------------------------------------------

    @property
    def running(self) -> bool:
        thread = PodcastsPipeline._thread
        return thread is not None and thread.is_alive()

    def due(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now()
        if now.time() < self.config.check_time():
            return False
        if self.running:
            return False
        return self.last_check_at is None or now.timestamp() - self.last_check_at >= CHECK_EVERY_S

    def maybe_start(self, now: Optional[datetime] = None) -> bool:
        """Starts a check when one is due; True when it did."""
        now = now or datetime.now()
        if not self.due(now):
            return False
        self.last_check_at = now.timestamp()
        return self._start(self.run_check, "PodcastsCheck")

    def _start(self, target, name: str, *args) -> bool:
        with self._lock:
            if self.running:
                info(f"Podcasts: a run is in progress, {name} not started")
                return False
            PodcastsPipeline._thread = threading.Thread(target=self._guarded, args=(target, *args), daemon=True, name=name)
            PodcastsPipeline._thread.start()
            return True

    def _guarded(self, target, *args):
        try:
            target(*args)
        except Exception as e:
            error(f"Podcasts: {target.__name__} failed: {type(e).__name__}: {e}")
            self._problem(f"{type(e).__name__}: {e}")
        finally:
            self.heartbeat(state="idle", last_error=self.last_problem)

    def _problem(self, problem: str):
        """Logs `problem` and tells the owner, once until it changes or a run goes well."""
        error(f"Podcasts: {problem}")
        if problem != self.last_problem:
            self._notify(f"Podcasts: {problem}")
        self.last_problem = problem

    # --- the check ------------------------------------------------------------------------

    def run_check(self) -> list[dict]:
        """One check of every show; the manifests of the episodes it finished."""
        shows = self.config.shows()
        if not shows:
            info("Podcasts: no shows configured — nothing to check")
            return []
        llm_settings = self.config.llm()
        if llm_settings.provider == PROVIDER_OPENAI and not llm_settings.api_key:
            self._problem("no OpenAI API key (podcasts.openai_api_key or OPENAI_API_KEY) — the episodes wait")
            return []
        done = []
        for show in shows:
            try:
                manifest = self.process_next(show)
            except source.ToolMissing as e:
                self._problem(str(e))
                return done
            if manifest:
                done.append(manifest)
        if not done:
            trace("Podcasts: check over, no new episode")
        return done

    def process_next(self, show: Show) -> Optional[dict]:
        library = self.library()
        state = self.load_state()
        found: dict = {}

        def is_processed(video_id: str, slug: str) -> bool:
            return video_id in state["processed"] or library.is_done(show, slug) \
                or state["failed_attempts"].get(video_id, 0) >= MAX_FAILED_ATTEMPTS

        def on_found(candidate: source.Candidate):
            found["candidate"] = candidate
            self._notify(f"🎧 {show.name}: new episode “{candidate.title}” — fetching and transcribing, "
                         f"the report follows.")

        def progress(step: str):
            candidate = found.get("candidate")
            self.heartbeat(state="running", show=show.name, slug=candidate.slug if candidate else None,
                           step=step, started_at=found.setdefault("started_at", time.time()))

        progress("checking the playlist")
        started = time.time()
        try:
            bundle = source.fetch_next(show.playlist_url, is_processed, self.bundles_root, show.slug_prefix,
                                       progress, on_found)
        except source.NoTranscriptAvailable as e:
            candidate = found["candidate"]
            note(f"Podcasts: {e} — {candidate.slug} is given up on")
            self.mark_processed(candidate.video_id)
            self._notify(f"{show.name}: “{candidate.title}” has no transcript — skipped.")
            return None
        except source.ToolMissing:
            raise
        except Exception as e:
            return self._failed(show, found.get("candidate"), f"fetch: {type(e).__name__}: {e}")
        if bundle is None:
            return None
        meta = report.load_meta(bundle)
        try:
            manifest = report.generate(bundle, show.output_folder / meta["slug"], self._llm_factory(self.config),
                                       self.config.artifacts(), show.name, progress)
        except (report.ReportError, ReportLLMError, OSError) as e:
            return self._failed(show, found.get("candidate"), f"report: {e}")
        self.mark_processed(meta["video_id"])
        self.last_problem = None
        timings = json.loads((show.output_folder / meta["slug"] / report.META).read_text(encoding="utf-8"))["timings"]
        total = time.time() - started
        info(f"Podcasts: {meta['slug']} finished in {describe_seconds(total)} — {describe_timings(timings)}")
        self._notify(f"✅ {show.name}: report ready for “{meta['title']}” in {show.output_folder / meta['slug']} "
                     f"({describe_seconds(total)}: {describe_timings(timings)}).")
        return manifest

    def _failed(self, show: Show, candidate: Optional[source.Candidate], problem: str) -> None:
        if candidate is None:
            self._problem(f"{show.name} check failed — {problem}")
            return None
        attempts = self.record_failure(candidate.video_id)
        if attempts >= MAX_FAILED_ATTEMPTS:
            self.mark_processed(candidate.video_id)
            self._notify(f"❌ {show.name}: “{candidate.title}” failed {attempts} times and is given up on — {problem}")
        else:
            self._notify(f"⚠️ {show.name}: “{candidate.title}” failed (attempt {attempts} of {MAX_FAILED_ATTEMPTS}, "
                         f"retried at the next check) — {problem}")
        error(f"Podcasts: {candidate.slug} failed, attempt {attempts}: {problem}")
        return None

    # --- on demand ------------------------------------------------------------------------

    def _show_of(self, slug: str) -> Optional[Show]:
        episode = self.library().episode(slug)
        return next((s for s in self.config.shows() if episode and s.name == episode.show), None)

    def start_regenerate(self, slug: str, kind: str) -> str:
        """Starts regenerating one artifact on a thread; what to tell the user."""
        library = self.library()
        episode = library.episode(slug)
        if episode is None:
            return f"podcast_regenerate failed: no episode {slug!r} — podcast_episodes lists them"
        if kind not in report.ARTIFACT_FILES:
            return f"podcast_regenerate failed: kind must be one of {', '.join(report.ARTIFACT_FILES)}"
        if episode.legacy or library.bundle(episode.slug) is None:
            return (f"podcast_regenerate failed: {episode.slug} has no transcript here (it predates this pipeline), "
                    f"so it cannot be regenerated")
        if not self._start(self._regenerate, "PodcastsRegenerate", episode.slug, kind):
            return "podcast_regenerate failed: a podcast run is in progress — try again when it is done"
        return f"Regenerating the {kind} of {episode.label()}; a message follows when it is done."

    def _regenerate(self, slug: str, kind: str):
        show = self._show_of(slug)
        bundle = self.library().bundle(slug)
        self.heartbeat(state="running", show=show.name, slug=slug, step=f"regenerating {kind}", started_at=time.time())
        try:
            report.generate(bundle, show.output_folder / slug, self._llm_factory(self.config), [kind], show.name,
                            lambda step: self.heartbeat(state="running", show=show.name, slug=slug, step=step),
                            regenerate=True)
        except (report.ReportError, ReportLLMError, OSError) as e:
            self._notify(f"❌ {show.name}: regenerating the {kind} of {slug} failed — {e}")
            return
        self._notify(f"✅ {show.name}: the {kind} of {slug} is regenerated.")

    def draft(self, slug: str, idea_no: int) -> str:
        """Expands idea `idea_no` of the episode's linkedin_ideas.md into a post, saved beside it."""
        library = self.library()
        episode = library.episode(slug)
        if episode is None:
            return f"podcast_draft failed: no episode {slug!r} — podcast_episodes lists them"
        ideas_path = episode.artifacts.get("linkedin_ideas")
        if ideas_path is None:
            return f"podcast_draft failed: {episode.slug} has no LinkedIn ideas"
        ideas = report.ideas_of(ideas_path.read_text(encoding="utf-8"))
        if idea_no not in ideas:
            return f"podcast_draft failed: {episode.slug} has ideas {', '.join(map(str, sorted(ideas))) or 'none'}, not {idea_no}"
        bundle = library.bundle(episode.slug)
        chapters = report.load_chapters(bundle) if bundle else []
        meta = report.load_meta(bundle) if bundle else {"title": episode.title, "video_url": episode.manifest.get("url", "")}
        try:
            post = report.draft_post(self._llm_factory(self.config), meta, ideas[idea_no], chapters)
        except ReportLLMError as e:
            return f"podcast_draft failed: {e}"
        path = episode.folder / f"linkedin_draft_{idea_no}.md"
        report.write_atomically(path, post + "\n")
        info(f"Podcasts: draft of idea {idea_no} of {episode.slug} saved to {path}")
        return f"Draft of idea {idea_no} of {episode.label()} (saved to {path}):\n\n{post}"

    @staticmethod
    def _send_to_owner(text: str):
        from assistant.Notifier import notifier
        try:
            notifier().send_message(text)
        except Exception as e:
            error(f"Podcasts: the message to the owner could not be sent: {type(e).__name__}: {e}")
