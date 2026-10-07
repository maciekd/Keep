"""Proactive toolkit loading: the toolkits a channel wants in the LLM's window before a
request from it starts (`ChannelProfile.preload_toolkits`, or the `preload_toolkits.<channel>`
config list in its place — voice wants `home`, with the home data, `music`, `tesla` and
`weather`, since "turn on the kitchen lamp", "turn off the music", "defrost the car" and
"will it rain" are the product and a `load_toolkit` round there costs a second or two on
every request), in front of them the channel's `ChannelProfile.fixed_toolkits`, which no
config list replaces (voice's `voice`, the spoken dialog style, which lives there and not in
the prompt, and `timers`, the spoken "set a timer" that every household asks for), plus
whatever the caller adds for the moment (`FirstRun.preload_toolkits` — the `setup` toolkit
while the FTUE lasts, on any channel). A toolkit whose integration is off
(`Toolkit.offered_by`, the IntegrationMonitor's verdict) is skipped: a house without a Tesla
preloads no Tesla.

Runs on the request thread right before the messages are assembled, after the history
window is trimmed the way `get_messages` is about to (so "in the window" means the same
thing to both). A toolkit already in the window (`MessageHistory.loaded_toolkits`) is left
alone; a missing one is persisted as a `command` row — the same text `load_toolkit` would
have returned, under a one-line preface that says nobody asked for it — so the request that
follows sees it and every later request finds it in the window until the history trims it
away. A toolkit in the window whose carried data has no known baseline
(`PromptManager.untracked_sections` — sent before a restart, or a pointer note ended the
tracking) gets its data again the same way, since the copy the LLM holds is stale.

One preload at a time: the iOS client sends a speculative request per partial transcript
("Turn", "Turn off", "Turn off the music" within 160 ms), each on its own handler thread,
and two of them checking the window before either had persisted both loaded `home` — two
7 KB rows the real request then inherited, straight past the window's high watermark
(2026-09-14). The check and the persist happen under `_lock`, and a request its successor
killed while it waited for the lock persists nothing: the successor does its own preload
and finds whatever is there.
"""
import threading
from typing import Callable

from assistant.ChannelProfile import ChannelProfile
from assistant.commands import Toolkits
from common.ConfigManager import config
from common.logging import info, trace

_lock = threading.Lock()


def preface(toolkit: Toolkits.Toolkit, channel: str) -> str:
    return (f"The {toolkit.name} toolkit is loaded ahead of this {channel} request, no load_toolkit needed "
            f"— its reference{' and the current home data' if toolkit.sections else ''} follow.")


def refresh_preface(toolkit: Toolkits.Toolkit, channel: str) -> str:
    return (f"The {toolkit.name} toolkit's reference is above in this conversation; its data may have changed "
            f"since it was sent, so the current data follows ahead of this {channel} request.")


def render_load(toolkit: Toolkits.Toolkit, prompt_manager, record: bool = True) -> str:
    """What `load_toolkit` sends for a toolkit that is not in the window: the reference,
    then the carried data (which also makes it the baseline for the delta notes, unless
    `record` is off — see PromptManager.carried_data). Without a PromptManager (tests) the
    placeholders stay and the data is left out."""
    substitute = prompt_manager.render_template if prompt_manager else (lambda text: text)
    text = Toolkits.render(toolkit, substitute)
    if toolkit.sections and prompt_manager is not None:
        if data := prompt_manager.carried_data(toolkit, record=record):
            text += "\n\n" + data
    return text


def preload(prompt_manager, profile: ChannelProfile, extra: tuple[str, ...] = (),
            is_killed: Callable[[], bool] | None = None) -> list[str]:
    """Put the profile's toolkits and *extra* in the window unless they are there already;
    returns the names loaded now. Nothing is loaded for a request `is_killed` reports as
    killed once the lock is held — a later request superseded it and preloads for itself."""
    configured = (config.get('preload_toolkits', {}) or {}).get(profile.name)
    channel_toolkits = profile.preload_toolkits if configured is None else tuple(configured)
    wanted = tuple(dict.fromkeys(profile.fixed_toolkits + channel_toolkits + tuple(extra)))
    if not wanted:
        trace(f"Toolkit preload: the {profile.name} channel preloads nothing and nothing else is wanted")
        return []
    with _lock:
        if is_killed is not None and is_killed():
            info(f"Toolkit preload: the {profile.name} request was killed before its turn at the window — "
                 f"nothing loaded, the request that superseded it preloads for itself")
            return []
        return _preload_locked(prompt_manager, profile, wanted, _monitor())


def _monitor():
    from assistant.integration.IntegrationMonitor import IntegrationMonitor
    return IntegrationMonitor.get_instance()


def _preload_locked(prompt_manager, profile: ChannelProfile, wanted: tuple[str, ...], monitor) -> list[str]:
    prompt_manager.message_history.trim_window()
    in_window = prompt_manager.message_history.loaded_toolkits()
    loaded = []
    for name in wanted:
        toolkit = Toolkits.by_name(name)
        if toolkit is None:
            info(f"Toolkit preload: the {profile.name} channel names an unknown toolkit '{name}' — skipped")
            continue
        if not toolkit.offered_by(monitor):
            info(f"Toolkit preload: {toolkit.name}'s integration is off — not preloaded for the {profile.name} channel")
            continue
        if toolkit.name in in_window:
            untracked = prompt_manager.untracked_sections(toolkit)
            if not untracked:
                trace(f"Toolkit preload: {toolkit.name} is already in the window — nothing to do")
                continue
            text = refresh_preface(toolkit, profile.name) + "\n\n" + prompt_manager.carried_data(toolkit)
            prompt_manager.add_command_result(text, cap=False)
            loaded.append(toolkit.name)
            info(f"Toolkit preload: {toolkit.name} is in the window but its {', '.join(untracked)} data has no "
                 f"known baseline — sent again for the {profile.name} request ({len(text)} chars)")
            continue
        text = preface(toolkit, profile.name) + "\n\n" + render_load(toolkit, prompt_manager)
        prompt_manager.add_command_result(text, cap=False)
        loaded.append(toolkit.name)
        info(f"Toolkit preload: {toolkit.name} was not in the window — loaded for the {profile.name} "
             f"request ({len(text)} chars)")
    return loaded
