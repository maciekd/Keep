import asyncio
import re
import time
from typing import Optional, Dict, Tuple, TYPE_CHECKING

from assistant.AssistantEvents import AssistantObserver
from common import Version
from common.SystemMonitor import SystemMonitor
from interfaces.console.tabs.HomeTab import HomeTab
from interfaces.console.tabs.DashboardTab import DashboardTab
from interfaces.console.tabs.LogTab import LogTab
from interfaces.console.tabs.PromptTab import PromptTab
from interfaces.console.tabs.VoiceTab import VoiceTab
from interfaces.console.tabs.BankTab import BankTab
from interfaces.console.tabs.ProceduresTab import ProceduresTab
from interfaces.console.tabs.WatchListTab import WatchListTab
from interfaces.console.tabs.VaultTab import VaultTab
from interfaces.console.tabs.CrystalTab import CrystalTab
from interfaces.console.framework.TUIScreen import TUIScreen
from interfaces.console.crystal.Scene import Scene
from interfaces.console.StartupIntro import StartupIntro
from interfaces.console.tabs import TabBar
from interfaces.console.tabs.IntegrationsGroup import IntegrationsGroup
from interfaces.console.tabs.TabGroup import TabGroup
from interfaces.console.tabs.TabNavigator import TabNavigator
from interfaces.console.widgets.ChatPane import ChatPane
from common.logging import info, trace

if TYPE_CHECKING:
    from interfaces.voice.microphones.speech_analysis.SpeechAnalyzer import SpeechAnalyzer


class TUIFloxyConsole(AssistantObserver):
    """Draws the tabbed TUI dashboard and routes keyboard input to the active tab.

    Tab selection and the Shift+arrow / Tab navigation live in TabNavigator; Memory, Integrations
    and Developer are TabGroups with their own row of sub-tabs. The Integrations row is not
    fixed: IntegrationsGroup shows the tab of each enabled integration that brings one (found in
    the integration folders) after the Catalog tab, which lists them all, enabled or not.

    The first frames after the first client connects are the startup intro (`StartupIntro`):
    the Crystal scene's flight over the whole screen and the title typing out beside the
    landed object, after which the Home tab keeps both as its header. The same scene is the
    Developer ▸ Crystal tab's.

    A process-wide singleton; it observes the Assistant for timing events and is created
    before the voice interface, which attaches its mics via attach_voice() later. The chat
    pane on the Home tab belongs to the ConsoleInterface's ConsoleTransport.
    """

    _instance: Optional['TUIFloxyConsole'] = None

    @classmethod
    def get_instance(cls, speech_analyzer=None, outpost_server=None, prompt_manager=None,
                     chat_pane=None) -> 'TUIFloxyConsole':
        if cls._instance is None:
            cls._instance = cls(speech_analyzer, outpost_server, prompt_manager, chat_pane)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        cls._instance = None

    def __init__(self, speech_analyzer: 'Optional[SpeechAnalyzer]', outpost_server, prompt_manager=None,
                 chat_pane=None):
        """`chat_pane` is the Home tab's chat (the screen side of the ConsoleTransport; a
        fresh, unconnected pane when None). The dashboard side of the console is fed through
        the AssistantObserver callbacks. speech_analyzer and outpost_server are None until
        the voice interface attaches (or forever, when voice is disabled) — the Developer ▸
        Voice tab and the dashboard then say so."""
        self.speech_analyzer = speech_analyzer
        self.time_to_silence_detect = -1  # endpointing share of STT (local outposts only)
        self.stt_trigger = ""             # gate that started the winning decode: "rms"/"vad"
        self.llm_from_cache = False       # LLM reply served from the response cache
        self.voice_from_cache = False     # audio served from the voice cache
        self.time_to_first_word = -1
        self.time_to_first_token = -1
        self.time_to_user_reply = -1
        self.time_to_first_sound_chunk = -1
        self.time_to_speaker = -1
        self.waiting_for_user = False
        self.current_noise_db = 0
        self.target_server_volume = 0
        self.outpost_latencies: Dict[str, Tuple[float, float]] = {}  # Store latency data: addr -> (rtt_ms, timestamp)
        self.outpost_server = outpost_server
        self.prompt_manager = prompt_manager

        self.system_total_memory_gb = 0.0
        self.system_used_memory_gb = 0.0
        self.process_memory_gb = 0.0
        self.process_cpu_percent = 0.0
        self.system_cpu_percent = 0.0
        self.fd_count = 0
        self.db_size_gb = 0.0
        self.voice_cache_size_gb = 0.0
        self.voice_cache_max_gb = 2.0
        self.photos_indexed = 0
        self.photos_total = 0

        self.system_monitor = SystemMonitor(tui_console=self, update_interval=0.33)
        self.system_monitor.start()

        self.scene = Scene()
        self.intro = StartupIntro(self.scene)
        self.navigator = TabNavigator([
            HomeTab(self, chat_pane if chat_pane is not None else ChatPane(), self.intro),
            DashboardTab(self),
            TabGroup("Memory", [
                BankTab(),
                ProceduresTab(),
                WatchListTab(),
            ]),
            VaultTab(),
            IntegrationsGroup(),
            TabGroup("Developer", [
                LogTab(),
                PromptTab(prompt_manager),
                VoiceTab(self),
                CrystalTab(self.scene),
            ]),
        ])
        self._force_clear_screen = False

    def draw_screen(self, frame: int, screen: TUIScreen):
        # A tab switch clears the terminal for real: the Log tab writes raw ANSI lines the
        # buffer diff knows nothing about, so they would otherwise linger under the next tab.
        if self._force_clear_screen:
            screen.clear_terminal()
            self._force_clear_screen = False

        if self.intro.draw(screen, frame):
            return
        self._draw_tab_bar(screen)
        self.navigator.active_tab.draw(screen, frame)

    def _draw_tab_bar(self, screen: TUIScreen):
        """Top tab bar on rows 1-2; a TabGroup draws its own sub-tab bar on row 4."""
        TabBar.fill_row(screen, 1, TabBar.TOP_BAR.fill)
        nav = self.navigator
        style = TabBar.TOP_BAR_GROUP_SELECTED if nav.group is not None else TabBar.TOP_BAR
        # The hint stands out until the user has navigated tabs once (remembered across restarts)
        TabBar.draw_tab_bar(screen, 2, [t.name for t in nav.tabs], nav.index, not nav.sub_focused, style,
                            hint="Tab or Shift-arrows to navigate tabs ",
                            hint_bright=not nav.navigated,
                            center=Version.label())

    # --- AssistantObserver -------------------------------------------------------------
    def on_first_token(self, seconds: float, from_cache: bool):
        self.time_to_first_token = seconds
        self.llm_from_cache = from_cache

    def on_reply_started(self, seconds_after_first_token: float):
        self.time_to_user_reply = seconds_after_first_token

    def on_first_sound_chunk(self, seconds: float, from_cache: bool):
        self.time_to_first_sound_chunk = seconds
        self.voice_from_cache = from_cache

    def on_speaker_output_started(self, seconds: float):
        self.time_to_speaker = seconds

    def on_waiting_for_user(self, waiting: bool):
        self.waiting_for_user = waiting

    def on_outpost_latency(self, addr, rtt_ms: float):
        self.outpost_latencies[str(addr)] = (rtt_ms, time.time())
        self.cleanup_stale_latencies()

    def on_server_volume(self, noise_db: float, target_volume: float):
        self.current_noise_db = noise_db
        self.target_server_volume = target_volume

    def on_wake_word_detected(self):
        self.reset()

    def on_stt_timing(self, time_to_silence_detect: float, time_to_first_word: float, stt_trigger: str):
        self.time_to_silence_detect = time_to_silence_detect
        self.time_to_first_word = time_to_first_word
        self.stt_trigger = stt_trigger

    def reset(self):
        self.time_to_first_token = -1
        self.time_to_user_reply = -1
        self.time_to_first_sound_chunk = -1
        self.time_to_speaker = -1

    def input_callback(self, key: str, client_addr: tuple):
        """Handle keyboard input from TUI clients"""
        if key.startswith(TUIScreen.PASTE_START):
            self._paste(key[len(TUIScreen.PASTE_START):-len(TUIScreen.PASTE_END)], client_addr)
            return
        # Parse the key directly - TUIServer now sends complete keystrokes
        parsed_key = self._parse_key(key)
        if not parsed_key:
            return
        if self.intro.playing:
            info(f"Client {client_addr} pressed a key during the startup intro - skipping to the logo.")
            self.intro.skip()
            return

        active_tab = self.navigator.active_tab
        # A tab taking typed text (search field, chat composer) gets every key first,
        # including '/' and '|'; only the Shift+arrow navigation is kept out of its way, and
        # Tab / Shift+Tab (the one navigation a phone keyboard can send) unless the tab is
        # revealing a masked field with it. Keys it does not consume (Ctrl+X, Ctrl+R) still
        # reach the global shortcuts below.
        content = active_tab.content_tab()
        navigation = parsed_key in self._NAVIGATION_KEYS
        if parsed_key in ('tab', 'shift_tab') and content.claims_tab:
            navigation = False
        if content.captures_text and not navigation:
            if active_tab.handle_input(parsed_key):
                return
        else:
            if self.navigator.handle_key(parsed_key):
                self._force_clear_screen = True
                return

            # Shift+Up/Down not taken by navigation scroll the tab by a page, as they always did
            if parsed_key == 'shift_up':
                parsed_key = 'pageup'
            elif parsed_key == 'shift_down':
                parsed_key = 'pagedown'
            if active_tab.handle_input(parsed_key):
                return

        # Handle global shortcuts (Ctrl+key)
        if parsed_key == '\x18':  # Ctrl+X
            info(f"Client {client_addr} pressed Ctrl+X - triggering wake word on all connected Outposts.")
            self._broadcast_outpost_command("listen")
            self.reset()
            if self.speech_analyzer is not None:
                self.speech_analyzer.clean()
        elif parsed_key == '\x12':  # Ctrl+R
            info(f"Client {client_addr} pressed Ctrl+R - restarting all connected Outposts.")
            self._broadcast_outpost_command("restart")

    def _paste(self, text: str, client_addr: tuple):
        """A bracketed paste from the terminal: one text, line breaks and all, to a tab that
        takes one (the Home tab's composer); typed key by key into any other, as every paste
        was before the markers were read — a line break is Enter there."""
        text = text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
        if self.intro.playing:
            info(f"Client {client_addr} pasted during the startup intro - skipping to the logo.")
            self.intro.skip()
            return
        if not text:
            trace(f"Client {client_addr} pasted nothing but line breaks, ignored")
            return
        active_tab = self.navigator.active_tab
        if active_tab.paste_text(text):
            return
        trace(f"Client {client_addr} pasted {len(text)} chars into {active_tab.content_tab().name}, "
              f"which takes no paste whole: typed key by key")
        for ch in text:
            self.input_callback(ch, client_addr)

    def attach_voice(self, speech_analyzer, outpost_server):
        """Called by the launcher once the voice interface is up (the console starts first
        so early log lines show)."""
        self.speech_analyzer = speech_analyzer
        self.outpost_server = outpost_server
        info("Console attached to the voice interface (Developer ▸ Voice tab, outpost shortcuts).")

    def _broadcast_outpost_command(self, command: str):
        if self.outpost_server is None:
            info(f"Cannot broadcast '{command}': voice interface disabled (no outpost server).")
            return
        loop = self.outpost_server.loop
        if loop is None or loop.is_closed():
            return
        try:
            future = asyncio.run_coroutine_threadsafe(self.outpost_server.broadcast_command(command), loop)
            future.result(timeout=10)
        except Exception as e:
            info(f"Failed to broadcast '{command}' to outposts: {e}")

    # The keys a text-taking tab never sees (Shift+Enter is text: a newline)
    _NAVIGATION_KEYS = ('shift_up', 'shift_down', 'shift_left', 'shift_right', 'tab', 'shift_tab')

    _NAMED_KEYS = {
        '\x1b[A': 'up', '\x1b[B': 'down', '\x1b[C': 'right', '\x1b[D': 'left',
        '\x1b[5~': 'pageup', '\x1b[6~': 'pagedown',
        # Shift+arrows: tab navigation (see TabNavigator), else page up/down
        '\x1b[1;2A': 'shift_up', '\x1b[1;2B': 'shift_down', '\x1b[1;2D': 'shift_left', '\x1b[1;2C': 'shift_right',
        '\x1b[H': 'home', '\x1b[1~': 'home', '\x1bOH': 'home',
        '\x1b[F': 'end', '\x1b[4~': 'end', '\x1bOF': 'end',
        '\x1b[3~': 'delete',
        '\t': 'tab', '\x1b[Z': 'shift_tab',
        # Word movement: Alt/Ctrl+arrows as CSI, or readline's ESC b / ESC f
        '\x1b[1;3D': 'alt_left', '\x1b[1;3C': 'alt_right', '\x1bb': 'alt_left', '\x1bf': 'alt_right',
        '\x1b[1;5D': 'ctrl_left', '\x1b[1;5C': 'ctrl_right',
        '\x1b\x7f': 'alt_backspace',
        # Alt+Enter as a legacy terminal sends it; the same newline as Shift+Enter
        '\x1b\r': 'shift_enter', '\x1b\n': 'shift_enter',
        '\x1b': 'escape',
    }
    # A key the terminal reports with its modifiers: `CSI code;mods u` (the kitty keyboard
    # protocol, which the TUI server turns on) or xterm's `CSI 27;mods;code ~`.
    _CSI_U = re.compile(r"\x1b\[(\d+)(?::\d*)*(?:;(\d+)(?::\d+)?)?u")
    _MODIFY_OTHER_KEYS = re.compile(r"\x1b\[27;(\d+);(\d+)~")
    # A legacy CSI key (arrows, Home/End, PgUp/PgDn, Delete) with its modifiers
    _CSI_MODIFIED = re.compile(r"\x1b\[(\d*);(\d+)(?::\d+)?([A-DHF~])")
    _MOD_SHIFT, _MOD_ALT, _MOD_CTRL = 1, 2, 4
    _MOD_LOCKS = 64 | 128   # Caps Lock, Num Lock: reported with the key, never part of its meaning

    @classmethod
    def _without_locks(cls, key: str) -> str:
        """`key` as it reads with Caps Lock and Num Lock off: `ESC[1;66A` (Shift+Up, Caps
        Lock on) is `ESC[1;2A`, `ESC[1;65A` is `ESC[A`, `ESC[3;65~` is `ESC[3~`."""
        match = cls._CSI_MODIFIED.fullmatch(key)
        if not match:
            return key
        number, mods, final = match.group(1), (int(match.group(2)) - 1) & ~cls._MOD_LOCKS, match.group(3)
        if mods:
            return f"\x1b[{number or 1};{mods + 1}{final}"
        return f"\x1b[{number}~" if final == '~' else f"\x1b[{final}"

    @classmethod
    def _parse_key(cls, key: str) -> str:
        """Name a keystroke or escape sequence; "" for sequences the TUI does not use."""
        if not key:
            return ""
        key = cls._without_locks(key)
        named = cls._NAMED_KEYS.get(key)
        if named is not None:
            return named
        if len(key) == 1:
            return key
        match = cls._CSI_U.fullmatch(key)
        if match:
            return cls._modified_key(int(match.group(1)), int(match.group(2) or 1))
        match = cls._MODIFY_OTHER_KEYS.fullmatch(key)
        if match:
            return cls._modified_key(int(match.group(2)), int(match.group(1)))
        # Bracketed-paste markers and any other sequence: nothing to do with them.
        return ""

    @classmethod
    def _modified_key(cls, code: int, mods: int) -> str:
        """The name of code point `code` pressed with `mods` (1 + the modifier bits), the
        same name the legacy bytes get: Ctrl+X is '\\x18' either way. Every modified Enter
        is 'shift_enter' — terminals differ in which of Shift, Alt and Ctrl they can report,
        and all of them mean a newline in the composer."""
        mods = (mods - 1) & ~cls._MOD_LOCKS
        shift, alt, ctrl = mods & cls._MOD_SHIFT, mods & cls._MOD_ALT, mods & cls._MOD_CTRL
        if code in (13, 57414):   # Enter, keypad Enter
            return 'shift_enter' if mods else '\r'
        if code == 27:
            return 'escape'
        if code == 9:
            return 'shift_tab' if shift else 'tab'
        if code == 127:
            return 'alt_backspace' if alt else '\x7f'
        if ctrl and 64 <= code <= 127:
            return chr(code & 0x1f)
        if alt:
            return {98: 'alt_left', 102: 'alt_right'}.get(code, "")
        ch = chr(code)
        return ch if ch.isprintable() else ""

    def cleanup_stale_latencies(self):
        current_time = time.time()
        stale_threshold = 8.0  # seconds

        # Create a list of addresses to remove to avoid modifying dict during iteration
        addresses_to_remove = [
            addr for addr, (_, timestamp) in self.outpost_latencies.items()
            if current_time - timestamp > stale_threshold
        ]

        for addr in addresses_to_remove:
            trace(f"Outpost at {addr} seems to have stopped.")
            del self.outpost_latencies[addr]
