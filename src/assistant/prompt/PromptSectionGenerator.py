import importlib
import pkgutil
import threading
import time
from abc import ABC, abstractmethod
from functools import lru_cache
from common.ConfigManager import config
from common.logging import error, trace, info

MODULE_SUFFIX = "SectionGenerator"


class PromptSectionGenerator(ABC):
    """Fetches one dynamic prompt section on its own thread and notifies PromptManager when it changes.

    Subclasses implement _fetch_content() and either set UPDATE_INTERVAL for plain polling or
    override _start_monitoring() for a custom update strategy. Failed refreshes keep the last
    good content marked stale rather than blanking the section.

    A generator in an integration's folder (`src/integrations/<folder>/<Name>SectionGenerator.py`)
    names its section with `key` and is found by `discover()`; the core's own generators are
    listed in PromptManager. The section reaches the LLM one of three ways: under its
    `heading` in the prompt's `{INTEGRATION_DATA}` block (the data of the integrations, one
    block the template places once), through a `{KEY}` placeholder of its own in the template
    (the core's memory, vault and the like), or carried by a toolkit (`sections:` in
    toolkit.md — `load_toolkit` sends it).

    A folder's generator runs only while the folder's integration is on; a core generator may
    tie itself to a contract the same way with `needs` (the calendar section to the calendar
    sources), or to a config.yaml key with `requires` (the voices section to the voice
    interface's `interfaces.voice.enabled`). PromptManager starts and stops them as the
    IntegrationMonitor's verdict changes; the config key is read at each verdict.
    """
    key: str = ""
    # The heading line the section stands under in the {INTEGRATION_DATA} block, hashes
    # included ("# CURRENT CAR INFO", "## Zones"); "" for a section placed otherwise.
    heading: str = ""
    # The text between the heading and the data — what the data is and how to use it.
    intro: str = ""
    # A core generator's contract: it runs while an enabled integration implements it.
    needs: type | None = None
    # A config.yaml key that must be true for the generator to run, read at each verdict.
    requires: str | None = None

    @classmethod
    def folder(cls) -> str | None:
        """The integration folder the generator belongs to; None for a core generator."""
        parts = cls.__module__.split(".")
        return parts[1] if len(parts) > 2 and parts[0] == "integrations" else None

    @classmethod
    def enabled_by(cls, monitor) -> bool:
        """Whether the IntegrationMonitor says the generator's integration is on: its folder's
        (`folder_enabled`), or the contract it `needs` (`contract_enabled`) — and not while the
        config key it `requires` is off. A generator tied to none of these is always on; one
        whose integration is not registered yet is not."""
        if cls.requires is not None and not config.get(cls.requires, False):
            return False
        if cls.folder() is not None:
            return monitor.folder_enabled(cls.folder()) is True
        if cls.needs is not None:
            return monitor.contract_enabled(cls.needs) is True
        return True

    def __init__(self, prompt_manager, section_name: str):
        self.prompt_manager = prompt_manager
        self.section_name = section_name
        self.content = ""
        self.last_update_time = 0
        self._refresh_failed_at = None
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            trace(f"Section '{self.section_name}' already started.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._start_monitoring_wrapper, name=f"prompt-{self.section_name}")
        self._thread.daemon = True
        self._thread.start()

    def _start_monitoring_wrapper(self):
        while not self._stop_event.is_set():
            try:
                self._start_monitoring()
                break
            except Exception as e:
                msg = str(e).replace("\n", "")
                info(f"Section '{self.section_name}' failed: {msg}. Retrying in 5 seconds...")
                self._wait(5)
                if not self._stop_event.is_set():
                    trace(f"Restarting monitoring for section '{self.section_name}'...")

    def stop(self):
        trace(f"Stopping section generator: {self.section_name}")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _wait(self, seconds: float) -> bool:
        """Sleep for `seconds` unless stop() is called first; True when it was. Monitoring loops
        wait with this, not time.sleep, so a stop() never has to wait out a poll interval."""
        return self._stop_event.wait(seconds)

    def get_content(self) -> str:
        """Get the current content of this section. This needs to be instant."""
        return self.content

    def _update_content(self):
        """
        Internal method to update content with error handling and retry logic.
        On failure after 3 retries, keeps the last good content marked as stale
        (or sets ERROR content if there is nothing to keep).
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                new_content = self._fetch_content()
                self._refresh_failed_at = None
                if new_content == self.content:
                    # skip if content unchanged
                    return
                self.content = new_content
                self.last_update_time = time.time()
                self.notify_update()
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    info(f"Failed to update section '{self.section_name}' (attempt {attempt + 1}/{max_retries}): {e}. Retrying in 1 second...")
                    time.sleep(1)
                else:
                    self._handle_fetch_failure(e, max_retries)

    def _handle_fetch_failure(self, e: Exception, max_retries: int):
        """Wiping good content over a transient outage would blind the LLM, so keep it and mark it stale instead."""
        if self.content and not self.content.startswith("ERROR:"):
            if self._refresh_failed_at is not None:
                error(f"Failed to update section '{self.section_name}' after {max_retries} attempts: {e}. Content already marked stale since {self._refresh_failed_at}.")
                return
            self._refresh_failed_at = time.strftime("%H:%M")
            error(f"Failed to update section '{self.section_name}' after {max_retries} attempts: {e}. Keeping last good content, marked stale.")
            self.content += f"\n[WARNING: refresh failed at {self._refresh_failed_at}, data above may be stale]"
        else:
            error(f"Failed to update section '{self.section_name}' after {max_retries} attempts: {e}. Setting ERROR content.")
            self.content = f"ERROR: Could not fetch section '{self.section_name}': {e}"
        self.last_update_time = time.time()
        self.notify_update()

    def notify_update(self):
        """Notify PromptManager that this section has been updated."""
        if self.prompt_manager:
            self.prompt_manager.on_section_updated(self.section_name)

    UPDATE_INTERVAL = None

    def _start_monitoring(self):
        """
        Start monitoring for updates. This method runs in a separate thread.
        Default implementation polls at UPDATE_INTERVAL seconds.
        Subclasses with non-standard strategies should override this.

        Must respect self._stop_event for graceful shutdown.
        """
        if self.UPDATE_INTERVAL is None:
            raise NotImplementedError("Subclass must set UPDATE_INTERVAL or override _start_monitoring")
        self._update_content()
        while not self._stop_event.is_set():
            self._wait(self.UPDATE_INTERVAL)
            if not self._stop_event.is_set():
                self._update_content()

    @abstractmethod
    def _fetch_content(self) -> str:
        """
        Fetch the actual content for this section.
        This method should return a string containing the section content.

        Returns:
            str: The content for this prompt section
        """
        pass


@lru_cache(maxsize=1)
def discover() -> tuple[type[PromptSectionGenerator], ...]:
    """The section generators of the `integrations` package, found once per process."""
    import integrations
    return find_in(integrations)


def find_in(package) -> tuple[type[PromptSectionGenerator], ...]:
    """Every PromptSectionGenerator subclass in a `*SectionGenerator` module of `package`, in
    the package's walk order. Only those modules are imported; one that fails to import is
    logged and left out, so a broken folder costs its own section and nothing else."""
    found: list[type[PromptSectionGenerator]] = []

    def skipped(fqn):
        error(f"Package {fqn} could not be walked for section generators, its section is left out")

    for _finder, fqn, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + ".", onerror=skipped):
        if ispkg or not fqn.endswith(MODULE_SUFFIX):
            continue
        try:
            module = importlib.import_module(fqn)
        except Exception as e:
            error(f"Section generator module {fqn} could not be imported, its section is left out: {type(e).__name__}: {e}")
            continue
        for attr in vars(module).values():
            if isinstance(attr, type) and issubclass(attr, PromptSectionGenerator) and attr is not PromptSectionGenerator \
                    and attr.__module__ == fqn and attr not in found:
                found.append(attr)
    trace(f"Section generators discovered in {package.__name__}: {', '.join(cls.__name__ for cls in found) or 'none'}")
    return tuple(found)
