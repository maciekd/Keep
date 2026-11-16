"""The camera contract: what the assistant asks of whatever owns cameras — Arlo and Ring today,
a local NVR such as UniFi Protect tomorrow. A provider is always an Integration too (name, config
keys, health); `fetch_camera_images` fans out over every enabled and configured one through
`active()` and never names a folder. The result shape is the LLM's: Anthropic content blocks, an
image block followed by a "Camera: <name>" text block per camera, a text block for a camera that
failed, nothing else — the one header line over all providers is the command's."""
from abc import ABC, abstractmethod
from typing import Iterable, Optional

from common.logging import info, trace


class CameraProvider(ABC):
    """What a camera integration implements."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def camera_names(self) -> list[str]:
        """The cameras the provider would fetch from (the ignored ones left out); [] when it
        cannot reach them."""
        ...

    @abstractmethod
    def fetch_snapshots(self, only: Optional[list[str]] = None) -> list:
        """A fresh still from every camera, or from the ones *only* names (see `select`), as
        content blocks. [] when no camera is selected. May raise when the provider is not
        connected: the command logs it and carries on with the other providers."""
        ...


def select(names: Iterable[str], ignore_keywords: Iterable[str] = (), only: Optional[Iterable[str]] = None) -> list[str]:
    """The cameras to fetch out of *names*: minus the ones a configured `ignore_cameras` keyword
    matches, narrowed to the ones an entry of *only* matches when the LLM asked for specific
    cameras. Both match case-insensitively on a substring, so "front" finds "Front Door"."""
    ignore = [k.lower() for k in ignore_keywords if k]
    wanted = [w.strip().lower() for w in (only or ()) if w and w.strip()]
    chosen = []
    for name in names:
        lower = name.lower()
        if any(k in lower for k in ignore):
            continue
        if wanted and not any(w in lower for w in wanted):
            continue
        chosen.append(name)
    return chosen


def active() -> list[CameraProvider]:
    """The providers the camera command fans out over: every registered CameraProvider whose
    config keys are set, in registration order. Resolved per call, so a restarted or newly
    configured integration is what the next command uses."""
    from assistant.integration.IntegrationMonitor import IntegrationMonitor
    ready = []
    for integration in IntegrationMonitor.get_instance().of_type(CameraProvider):
        missing = integration.missing_config_keys()
        if missing:
            trace(f"Cameras: {integration.name} is enabled but not configured ({', '.join(missing)}), skipped")
            continue
        ready.append(integration)
    info(f"Cameras: {len(ready)} provider(s) ready" + (f" ({', '.join(p.name for p in ready)})" if ready else ""))
    return ready


def no_provider_hint() -> str:
    """Why `active()` came back empty, for the LLM: the keys of an unconfigured provider, the
    error of one whose constructor failed, the enable line for a disabled one, or "still
    starting" while none is registered (the launcher has not reached the integrations yet)."""
    from assistant.integration.IntegrationMonitor import IntegrationMonitor, DisabledIntegration, UnavailableIntegration
    monitor = IntegrationMonitor.get_instance()
    hints = []
    for integration in monitor.of_type(CameraProvider):
        hints.append(f"{integration.name} is enabled but not configured: set "
                     f"{', '.join(integration.missing_config_keys())} (config_help / config_request).")
    for placeholder in monitor.of_type(DisabledIntegration):
        if placeholder.can_enable and issubclass(placeholder.cls, CameraProvider):
            hints.append(f'{placeholder.name} is disabled: enable_integration("{placeholder.name}").')
    for placeholder in monitor.of_type(UnavailableIntegration):
        if issubclass(placeholder.cls, CameraProvider):
            hints.append(f"{placeholder.name} could not be started: {placeholder.last_error} (restart_integration once fixed).")
    if not hints:
        trace("Cameras: no provider registered yet — the integrations are still starting")
        return STARTING_HINT
    hint = " ".join(hints)
    trace(f"Cameras: no provider ready — {hint}")
    return hint


STARTING_HINT = "The cameras are still starting up — try again in a few seconds."
