"""Discovers local microphones and runs one LocalOutpost per mic.

Started at server boot. Warms up the shared Whisper worker, enumerates input
devices (skipping virtual/aggregate ones and anything in the configured exclude
list), and spins up an outpost per real mic. Also routes the server's broadcast
commands (stop / listen / wakeword) to every outpost so they mute while Floxy
speaks and resume afterwards.

mlx_whisper is Apple-Silicon only; if it (or any audio dependency) is unavailable
the manager logs why and stays disabled rather than taking the server down.
"""

import gc
import os
import re
import shutil
import subprocess
import threading
import time
import traceback

from .constants import (
    DEFAULT_MODEL_REPO, VAD_BACKEND, VAD_THRESHOLD, VIRTUAL_MIC_NAME_HINTS, MAX_OUTPOSTS,
    RECOVERY_TICK_INTERVAL, RECOVERY_STALE_TIMEOUT, RECOVERY_OFFBUS_GRACE, USB_POWER_CYCLE_COOLDOWN,
    USB_POWER_CYCLE_OFF_SECONDS, USB_POWER_CYCLE_MAX_ATTEMPTS, UHUBCTL_PATHS,
    RESOURCE_MONITOR_INTERVAL, SCRIBE_ENABLED, KEEPWARM_IDLE_SECONDS,
)
from .LocalOutpost import LocalOutpost
from .OutpostStreamClient import OutpostStreamClient
from .ScribeWorker import ScribeWorker
from .scribe_status import set_scribe_running
from .TextLanguageDetector import TextLanguageDetector
from .WhisperWorker import WhisperWorker, model_cached, model_download_size
from assistant.SystemNote import system_note
from common.logging import info, trace, error, note, fine

# uhubctl output parsing (one device per powered hub port):
#   Current status for hub 8-3 [05ac:8009 Apple USB2 Hub 0000A00, USB 2.10, 4 ports, ppps]
#     Port 4: 0503 power highspeed enable connect [2752:001c miniDSP VocalFusion Spk (UAC2.0)]
_HUB_RE = re.compile(r'^Current status for hub (\S+) \[(.*)\]')
_PORT_RE = re.compile(r'^\s*Port (\d+):')
_DEV_RE = re.compile(r'\[[0-9a-fA-F]{4}:[0-9a-fA-F]{4}\s+(.+)\]')


def build_lang_routing(languages):
    """Derive the local-outpost language routing from the configured BCP-47 locales.

    Returns (default_lang, lang_by_subtag, whisper_language):
    - default_lang: the primary locale (first), used as the /stream fallback;
    - lang_by_subtag: {subtag: full-locale} (e.g. {"en": "en-US", "sv": "sv-SE"}) that
      turns Whisper's detected language into a /stream locale (and gates out-of-set ones);
    - whisper_language: a single subtag to force on Whisper when only one language is
      configured (best accuracy), or None to auto-detect per utterance when several are.
    """
    languages = list(languages) if languages else ["en-US"]
    lang_by_subtag = {}
    for tag in languages:
        lang_by_subtag.setdefault(tag.split("-")[0].lower(), tag)
    whisper_language = next(iter(lang_by_subtag)) if len(lang_by_subtag) == 1 else None
    return languages[0], lang_by_subtag, whisper_language


def filter_input_devices(devices, exclude_names):
    """Pure helper: pick real input mics from a sounddevice query_devices() list.

    Skips devices with no input channels, known virtual/aggregate/loopback devices
    (by name substring), and anything whose name contains a configured exclude hint.
    Returns a list of (index, name).
    """
    if isinstance(exclude_names, str):
        # `exclude_mics: Bose QC45` written by hand: one hint, not one per character
        exclude_names = [exclude_names]
    exclude = tuple(e.lower() for e in (exclude_names or []))
    mics = []
    for idx, dev in enumerate(devices):
        name = dev.get("name", f"device {idx}")
        if dev.get("max_input_channels", 0) <= 0:
            continue
        lname = name.lower()
        hint = next((h for h in VIRTUAL_MIC_NAME_HINTS if h in lname), None)
        if hint:
            trace(f"Skipping virtual input device #{idx} '{name}' (matched '{hint}').")
            continue
        excl = next((e for e in exclude if e in lname), None)
        if excl:
            trace(f"Skipping excluded input device #{idx} '{name}' (matched config exclude '{excl}').")
            continue
        mics.append((idx, name))
    return mics


def disambiguate_mic_names(mics):
    """Append ' #N' to mics that share a name (e.g. two 'Studio Display Microphone')
    so logs and outpost labels are unambiguous. Unique names are left untouched.
    Keeps the real device index; only the display name changes."""
    from collections import Counter
    counts = Counter(name for _, name in mics)
    seen = {}
    result = []
    for idx, name in mics:
        if counts[name] > 1:
            seen[name] = seen.get(name, 0) + 1
            result.append((idx, f"{name} #{seen[name]}"))
        else:
            result.append((idx, name))
    return result


class LocalOutpostManager:
    """Discovers the local microphones and runs one LocalOutpost per mic.

    Also routes the server's broadcast commands to every outpost, and supervises mic
    recovery (stream re-attach, USB power-cycling) and the resource monitor.
    """

    # The one live manager of this process (set on construction, None when local
    # outposts are disabled) — lets the Stats tab read per-mic gate state without
    # threading a reference through the console.
    instance = None

    def __init__(self, port, model_repo=DEFAULT_MODEL_REPO, languages=None,
                 exclude_mics=None, vad_backend=VAD_BACKEND, vad_threshold=VAD_THRESHOLD,
                 auto_recovery=True, usb_power_cycle=True):
        self.port = port
        self.model_repo = model_repo
        self.languages = list(languages) if languages else ["en-US"]
        self.default_lang, self.lang_by_subtag, self.whisper_language = build_lang_routing(self.languages)
        # Second-guesses Whisper's flaky per-clip language label using the decoded text, and
        # vetoes wake-word fires whose transcript is confidently a non-command language
        # (overheard household speech). for_outpost loads every classifier-known language as a
        # candidate so is_foreign works even in a single-accepted-language (English-only)
        # deployment, where a plain detector would be inactive.
        self.lang_detector = TextLanguageDetector.for_outpost(set(self.lang_by_subtag))
        self.exclude_mics = exclude_mics or []
        self.vad_backend = vad_backend
        self.vad_threshold = vad_threshold
        self._worker = None
        self._client = None
        self._scribe = None
        self.outposts = []
        self._active = False

        # ── auto-recovery state ──
        self.auto_recovery = auto_recovery
        self.usb_power_cycle = usb_power_cycle
        self._lock = threading.RLock()      # guards self.outposts (recovery vs. command threads)
        self._expected_names = []           # mic names we want running (persists across drops)
        self._port_by_mic = {}              # mic name -> (hub, port), learned from uhubctl while present
        self._last_cycle_at = {}            # mic name -> time.time() of last power-cycle (cooldown)
        self._cycle_failures = {}           # mic name -> consecutive cycles with no re-enumeration
        self._gave_up_cycling = set()       # mics past USB_POWER_CYCLE_MAX_ATTEMPTS (re-armed on bus return)
        self._offline_names = set()         # mics currently flagged offline (for transition logging)
        self._offbus_since = {}             # mic name -> time.time() it first went missing from the USB scan
        self._pending_reattach = set()      # mics that left the bus and need a stream re-attach once back
        self._uhubctl = None                # resolved uhubctl path, or None if power-cycling is off
        self._recovery_thread = None
        self._stop_recovery = threading.Event()  # set on stop(); also halts the resource monitor

        # ── resource monitor (leak hunt) ──
        self._monitor_thread = None
        self._last_rss_mb = None             # previous snapshot's RSS, for the per-tick delta

        LocalOutpostManager.instance = self

    @staticmethod
    def _note_scribe_unavailable():
        """Tell the household and the LLM why voice is degraded; a log line alone leaves the
        assistant unaware that the wake word cannot be verified."""
        from common.ConfigManager import config
        if (config.get("api_keys", {}) or {}).get("elevenlabs"):
            text = "ElevenLabs Scribe is unavailable for the voice interface. Voice will not work now."
        else:
            text = ("ElevenLabs API key is needed for the voice interface (api_keys.elevenlabs). "
                    "Voice will not work now.")
        system_note("voice", text)

    def _announce_download_start(self):
        """The first start on a machine downloads the model inside warm_up with no visible
        progress; without a note the household sees a silent server for minutes."""
        size = model_download_size(self.model_repo)
        size_text = f"{size / 1e9:.1f} GB" if size else "several GB"
        system_note("voice", f"Downloading the Whisper speech-to-text model '{self.model_repo}' "
                             f"from Hugging Face ({size_text}, a one-time download). The local "
                             f"microphones are not listening until it is done.")

    def start(self):
        downloading = not model_cached(self.model_repo)
        if downloading:
            self._announce_download_start()
        t0 = time.time()
        try:
            self._worker = WhisperWorker(self.model_repo, self.whisper_language,
                                         allowed_languages=set(self.lang_by_subtag))
            self._worker.warm_up()
        except Exception as e:
            error(f"Local outposts disabled — Whisper unavailable on this platform: {e}")
            if downloading:
                system_note("voice", f"The Whisper speech-to-text model download failed ({e}). "
                                     f"Voice will not work now; it is retried at the next start.")
            return False
        if downloading:
            system_note("voice", f"Whisper speech-to-text model downloaded and loaded "
                                 f"({(time.time() - t0) / 60:.1f} min).")

        mics = self._discover_mics()
        if not mics:
            error("No local microphones found — no local outposts will run.")
            return False

        self._worker.start()
        self._client = OutpostStreamClient(self.port)
        if SCRIBE_ENABLED:
            try:
                self._scribe = ScribeWorker()
                info("Scribe v2 parallel-vote STT enabled (runs alongside Whisper).")
            except Exception as e:
                # Missing key / unavailable deps must not take the Whisper outposts down.
                error(f"Scribe parallel-vote STT unavailable, continuing Whisper-only: {e}")
                self._scribe = None
                self._note_scribe_unavailable()
        # Publish actual availability so the wake-word verdict gate only holds replies
        # when a Scribe verdict can really arrive (Whisper-only fallback: never hold).
        set_scribe_running(self._scribe is not None)
        for idx, name in mics:
            self._start_outpost(idx, name)

        if not self.outposts:
            info("No local outposts could be started.")
            return False

        self._active = True
        self._expected_names = [name for _, name in mics]
        whisper_mode = f"forced '{self.whisper_language}'" if self.whisper_language else \
            f"auto-detect {sorted(self.lang_by_subtag)}"
        note(f"Local outposts running on {len(self.outposts)} mic(s), "
             f"languages={self.languages} (Whisper {whisper_mode}).")
        system_note("voice", self._running_note(), push=False)
        self._start_recovery_loop()
        self._start_monitor_loop()
        return True

    def _running_note(self) -> str:
        """The household-facing wording of the start-up log line: a system row so the chat
        history and the LLM know voice is listening, on which mics and in what language.
        History only — every start says it, so it is not pushed to the chats."""
        with self._lock:
            names = ", ".join(o.device_name for o in self.outposts)
        count = len(self.outposts)
        mics = f"{count} local microphone{'s' if count != 1 else ''} ({names})"
        if len(self.languages) == 1:
            language = f"Language: {self.languages[0]}."
        else:
            language = f"Languages: {', '.join(self.languages)} (auto-detected per utterance)."
        return f"Voice is ready — listening on {mics}. {language}"

    def _start_outpost(self, idx, name):
        """Build, start, and register one outpost. Returns it, or None on failure.
        Shared by initial boot and the recovery loop's re-attach."""
        outpost = LocalOutpost(idx, name, self._worker, self._client,
                               self.default_lang, self.lang_by_subtag,
                               lang_detector=self.lang_detector, scribe_worker=self._scribe,
                               vad_backend=self.vad_backend, vad_threshold=self.vad_threshold)
        try:
            outpost.start()
        except Exception as e:
            error(f"Failed to start local outpost for mic #{idx} '{name}': {e}")
            from .AudioDiagnostics import capture_open_failure
            capture_open_failure(name, name, e)
            return None
        with self._lock:
            self.outposts.append(outpost)
        return outpost

    def _discover_mics(self):
        import sounddevice as sd
        devices = sd.query_devices()
        mics = disambiguate_mic_names(filter_input_devices(devices, self.exclude_mics))
        for idx, name in mics:
            info(f"Discovered local microphone #{idx}: {name}")
        if MAX_OUTPOSTS is not None and len(mics) > MAX_OUTPOSTS:
            kept = mics[:MAX_OUTPOSTS]
            skipped = ", ".join(name for _, name in mics[MAX_OUTPOSTS:])
            trace(f"MAX_OUTPOSTS={MAX_OUTPOSTS}: using {', '.join(n for _, n in kept)}; "
                 f"skipping remaining mic(s): {skipped}")
            return kept
        return mics

    def on_command(self, command):
        """Route a broadcast command to every outpost. Called from the OutpostServer
        event loop thread; the outpost methods only flip thread-safe flags."""
        if not self._active:
            return
        with self._lock:
            outposts = list(self.outposts)
        if command == "stop":
            for o in outposts:
                o.on_mute()
        elif command == "listen":
            for o in outposts:
                o.on_listen()
        elif command == "wakeword":
            for o in outposts:
                o.on_wakeword()
        elif command == "resume":
            for o in outposts:
                o.on_resume()
        else:
            trace(f"Local outposts ignoring unknown command: {command}")

    def stop(self):
        self._stop_recovery.set()
        with self._lock:
            outposts = list(self.outposts)
            self.outposts.clear()
        for o in outposts:
            o.stop()
        if self._worker is not None:
            self._worker.stop()
        if self._scribe is not None:
            self._scribe.shutdown()
        set_scribe_running(False)
        if self._client is not None:
            self._client.shutdown()
        self._active = False

    # ── resource monitor (leak hunt) ──────────────────────────────────────────
    def _start_monitor_loop(self):
        """Launch the background watcher that logs one 'resmon' line of process/audio
        resource metrics per tick. It exists to reveal what grows over uptime as the
        capture-drop rate ramps (RSS, thread count, GC backlog, Whisper queue/spec-result
        counts, per-mic capture-queue depth) — grep 'resmon' to trend it against drops."""
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, name="resmon", daemon=True)
        self._monitor_thread.start()
        info(f"Resource monitor running (snapshot every {RESOURCE_MONITOR_INTERVAL:.0f}s) — "
             f"grep 'resmon' to trend RSS/threads/queues against capture drops.")

    def _monitor_loop(self):
        while not self._stop_recovery.wait(RESOURCE_MONITOR_INTERVAL):
            try:
                self._log_resource_snapshot()
            except Exception as e:
                error(f"Resource monitor tick failed: {e}")
                trace(traceback.format_exc())

    def _log_resource_snapshot(self):
        """Emit one compact line of process + audio-pipeline metrics. Cheap reads only
        (qsize/len/RSS) so the monitor itself never stalls the capture callback it watches."""
        try:
            import psutil
            rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            rss_mb = -1.0
        delta = ""
        if rss_mb >= 0:
            if self._last_rss_mb is not None:
                delta = f"({rss_mb - self._last_rss_mb:+.0f})"
            self._last_rss_mb = rss_mb
        g0, g1, g2 = gc.get_count()
        if self._worker is not None:
            wm = self._worker.metrics()
            worker = (f"jobs={wm['jobs']} spec_results={wm['spec_results']} "
                      f"cancelled={wm['cancelled']} inflight={wm['inflight']} "
                      f"spec_total={wm['spec_total']}")
        else:
            worker = "none"
        with self._lock:
            outposts = list(self.outposts)
        mics = " | ".join(
            f"{m['name']}: q={m['q']} pend={m['pending_drops']} "
            f"drops={m['drops']}/{m['lost_s']:.1f}s alive={m['alive']}"
            for m in (o.metrics() for o in outposts)) or "no mics"
        fine(f"resmon rss={rss_mb:.0f}MB{delta} threads={threading.active_count()} "
             f"gc=({g0},{g1},{g2}) garbage={len(gc.garbage)} | worker[{worker}] | {mics}")

    # ── auto-recovery ─────────────────────────────────────────────────────────
    def _start_recovery_loop(self):
        """Launch the background ticker. It always runs — the GPU keep-warm tick lives
        here too and must not die with the recovery feature; auto_recovery only gates
        the mic-watchdog part of each pass."""
        if not self.auto_recovery:
            info("Local-outpost auto-recovery disabled (local_outposts.auto_recovery is "
                 "false) — background ticker still runs for the GPU keep-warm.")
        else:
            self._uhubctl = self._resolve_uhubctl() if self.usb_power_cycle else None
            if self.usb_power_cycle and not self._uhubctl:
                note("Auto-recovery: uhubctl not found — USB power-cycling disabled (install with "
                     "'brew install uhubctl'). Offline mics will still be detected and re-attached.")
        self._recovery_thread = threading.Thread(
            target=self._recovery_loop, name="LocalOutpostRecovery", daemon=True)
        self._recovery_thread.start()
        if self.auto_recovery:
            trace(f"Auto-recovery loop running (check every {RECOVERY_TICK_INTERVAL:.0f}s, "
                  f"offline after {RECOVERY_STALE_TIMEOUT:.0f}s silent; "
                  f"USB power-cycle {'on' if self._uhubctl else 'off'}).")

    def _resolve_uhubctl(self):
        for p in UHUBCTL_PATHS:
            if "/" in p:
                if os.path.exists(p):
                    return p
            else:
                found = shutil.which(p)
                if found:
                    return found
        return None

    def _recovery_loop(self):
        while not self._stop_recovery.wait(RECOVERY_TICK_INTERVAL):
            try:
                self._keep_warm_tick()
                if self.auto_recovery:
                    self._recovery_tick()
            except Exception as e:
                error(f"Auto-recovery tick failed: {e}")
                trace(traceback.format_exc())

    def noise_floors(self):
        """[(mic label, floor RMS, threshold RMS, live 1s-min RMS)] from each outpost's
        adaptive energy gate, for the Stats tab's noise-floor panel. Floor/live are
        None until first samples land; outposts not yet started (no gate) are skipped."""
        with self._lock:
            outposts = list(self.outposts)
        floors = []
        for o in outposts:
            gate = getattr(o, "_energy_vad", None)
            if gate is not None:
                floors.append((o.label, gate.noise_floor, gate.threshold,
                               getattr(gate, "raise_min", None)))
        return floors

    def _keep_warm_tick(self):
        """Hold the GPU at its warm clock through quiet periods: after
        KEEPWARM_IDLE_SECONDS without a decode, submit a tiny silence decode so the
        first real query of the next conversation doesn't pay Metal's ~2.2–2.9x
        idle-decay penalty (see KEEPWARM_IDLE_SECONDS in constants). Never while a
        mic is mid-utterance — a real decode is imminent then and must not queue
        behind a keep-warm that just started (a warm keep-warm finishes ~2s before
        any utterance's decode is submitted, so this gate closes the only race)."""
        if self._worker is None or self._worker.idle_seconds() < KEEPWARM_IDLE_SECONDS:
            return
        with self._lock:
            if any(o.capturing_speech for o in self.outposts):
                return
        self._worker.submit_keepwarm()

    def _recovery_tick(self):
        """One watchdog pass: find mics that have gone offline, power-cycle any that have
        dropped off the USB bus, and re-attach any that are back on the bus."""
        usb = self._scan_usb_ports()       # {mic name: (hub, port)} on the bus now; None if scan unavailable
        if usb is not None:
            self._learn_ports(usb)
            # A mic on the bus proves its last cycle (or a replug) worked — re-arm cycling.
            for name in usb:
                if name in self._gave_up_cycling:
                    note(f"Auto-recovery: '{name}' is back on the USB bus — power-cycling re-armed.")
                    self._gave_up_cycling.discard(name)
                self._cycle_failures.pop(name, None)

        attention = []                     # [(name, outpost_or_None, reason)]
        for name in self._expected_names:
            o = self._find_outpost(name)
            off_bus = self._is_off_bus(name, usb)
            if o is None or not o.is_alive():
                attention.append((name, o, "no live outpost"))
            elif o.seconds_since_last_frame() > RECOVERY_STALE_TIMEOUT:
                attention.append((name, o, f"silent for {o.seconds_since_last_frame():.0f}s"))
            elif off_bus:
                # Frames still flow but the device left the bus: macOS rerouted the open
                # stream to another input device, so staleness will never fire. The stream
                # must be re-attached once the device is back, hence the pending flag.
                self._pending_reattach.add(name)
                attention.append((name, o, "gone from USB bus though frames still arrive "
                                           "(stream rerouted to another input device?)"))
            elif name in self._pending_reattach:
                attention.append((name, o, "back on USB bus — stream re-attach pending"))

        attention_names = {n for n, _, _ in attention}
        for name in list(self._offline_names):
            if name not in attention_names:
                note(f"Auto-recovery: mic '{name}' is back online.")
                self._offline_names.discard(name)
        if not attention:
            return

        reattach_needed = False
        for name, o, reason in attention:
            on_bus = (name in usb) if usb is not None else None
            if name not in self._offline_names:
                note(f"Auto-recovery: mic '{name}' needs attention ({reason}; on USB bus: "
                     f"{'unknown' if on_bus is None else 'yes' if on_bus else 'no'}).")
                self._offline_names.add(name)
            if on_bus is False:
                # Device has dropped off USB — a power-cycle re-enumerates it; we re-attach
                # on a later tick once it reappears on the bus.
                self._maybe_power_cycle(name)
            elif name in self._pending_reattach:
                # Back on the bus after a drop — its stream still points at the wrong
                # device, so re-attach even though frames look healthy.
                reattach_needed = True
            elif o is not None and o.is_alive() and on_bus and self._maybe_power_cycle(name):
                # On the bus but our stream is deaf — the device itself is wedged; the
                # power-cycle resets it and we re-attach once it re-enumerates.
                pass
            else:
                # On the bus (or bus state unknown) and just needs PortAudio to (re)open it.
                reattach_needed = True

        if reattach_needed:
            self._reattach()

    def _is_off_bus(self, name, usb):
        """True when a mic with a learned USB port has been missing from the bus scan for
        at least RECOVERY_OFFBUS_GRACE (debounces re-enumeration blips). Only learned mics
        are judged — one on a non-switchable hub is invisible to uhubctl by design — and an
        unavailable scan (usb=None) judges nothing."""
        if usb is None or name not in self._port_by_mic:
            return False
        if name in usb:
            self._offbus_since.pop(name, None)
            return False
        since = self._offbus_since.setdefault(name, time.time())
        return time.time() - since >= RECOVERY_OFFBUS_GRACE

    def _find_outpost(self, name):
        with self._lock:
            return next((o for o in self.outposts if o.device_name == name), None)

    def _scan_usb_ports(self):
        """Run uhubctl and return {expected mic name: (hub, port)} for mics currently
        present on a power-switchable (ppps) hub, or None when the scan can't run (no
        uhubctl, or it failed) — callers must treat None as "bus state unknown", not
        "everything is unplugged". Reads USB directly, so unlike PortAudio it reflects
        hot-plug changes immediately and never touches the audio subsystem."""
        if not self._uhubctl:
            return None
        try:
            r = subprocess.run([self._uhubctl], capture_output=True, text=True, timeout=10)
        except Exception as e:
            trace(f"Auto-recovery: uhubctl scan failed: {e}")
            return None
        hub, ppps, found = None, False, {}     # found: device label -> (hub, port)
        for line in r.stdout.splitlines():
            m = _HUB_RE.match(line)
            if m:
                hub, ppps = m.group(1), ("ppps" in m.group(2))
                continue
            pm = _PORT_RE.match(line)
            if pm and hub and ppps:
                bm = _DEV_RE.search(line)
                if bm:
                    found[bm.group(1).strip()] = (hub, int(pm.group(1)))
        result = {}
        for name in self._expected_names:
            hp = next((hp for desc, hp in found.items() if name.lower() in desc.lower()), None)
            if hp:
                result[name] = hp
        return result

    def _learn_ports(self, usb):
        """Cache each present mic's USB (hub, port) so a later drop — when uhubctl can no
        longer see the device by name — can still power-cycle the right port."""
        for name, hp in usb.items():
            if self._port_by_mic.get(name) != hp:
                self._port_by_mic[name] = hp
                trace(f"Auto-recovery: learned USB port for '{name}' → hub {hp[0]} port {hp[1]}.")

    def _maybe_power_cycle(self, name):
        """Power-cycle the mic's learned USB port via uhubctl (cooldown-guarded). Returns
        True if a cycle was actually issued, False if it couldn't (no uhubctl/port/cooldown,
        or cycling has been given up on as ineffective for this drop)."""
        if not self._uhubctl or name in self._gave_up_cycling:
            return False
        port = self._port_by_mic.get(name)
        if not port:
            note(f"Auto-recovery: '{name}' is offline but its USB port was never learned "
                 f"(not on a power-switchable hub?) — cannot power-cycle.")
            return False
        now = time.time()
        if now - self._last_cycle_at.get(name, 0.0) < USB_POWER_CYCLE_COOLDOWN:
            return False
        self._last_cycle_at[name] = now
        hub, p = port
        note(f"Auto-recovery: power-cycling USB port for '{name}' (hub {hub}, port {p}) via uhubctl…")
        try:
            r = subprocess.run(
                [self._uhubctl, "-a", "cycle", "-l", hub, "-p", str(p),
                 "-d", str(USB_POWER_CYCLE_OFF_SECONDS)],
                capture_output=True, text=True, timeout=30)
        except Exception as e:
            error(f"Auto-recovery: uhubctl power-cycle for '{name}' failed to run: {e}")
            return False
        if r.returncode == 0:
            fails = self._cycle_failures.get(name, 0) + 1
            self._cycle_failures[name] = fails
            if fails >= USB_POWER_CYCLE_MAX_ATTEMPTS:
                self._gave_up_cycling.add(name)
                error(f"Auto-recovery: '{name}' has not re-enumerated after {fails} consecutive "
                      f"power-cycles — its firmware is likely hung and this hub does not cut "
                      f"VBUS, so cycling cannot revive it. NEEDS A PHYSICAL REPLUG (or reboot). "
                      f"Giving up on cycling; will re-attach automatically if it reappears.")
            else:
                note(f"Auto-recovery: power-cycle issued for '{name}' (attempt {fails}/"
                     f"{USB_POWER_CYCLE_MAX_ATTEMPTS}) — waiting for it to re-enumerate "
                     f"on the USB bus.")
            return True
        error(f"Auto-recovery: uhubctl returned {r.returncode} for '{name}': "
              f"{(r.stderr or r.stdout).strip()[:200]}")
        return False

    def _reattach(self):
        """Re-initialize PortAudio (its device list is cached for the process lifetime) and
        restart every expected mic. This tears down all open audio streams, so it's deferred
        while Floxy is speaking; the loop retries on the next tick."""
        if self._server_busy():
            trace("Auto-recovery: deferring PortAudio refresh — Floxy is speaking.")
            return
        import sounddevice as sd
        note("Auto-recovery: refreshing audio subsystem (stopping mics, re-initializing PortAudio).")
        with self._lock:
            outposts = list(self.outposts)
            self.outposts.clear()
        for o in outposts:
            try:
                o.stop()
            except Exception:
                pass
        try:
            sd._terminate()
            sd._initialize()
        except Exception as e:
            error(f"Auto-recovery: PortAudio re-init failed: {e}")
        mics = self._discover_mics()
        present = {name for _, name in mics}
        for name in present:
            if name not in self._expected_names:
                self._expected_names.append(name)
        for idx, name in mics:
            if self._start_outpost(idx, name):
                self._pending_reattach.discard(name)
                note(f"Auto-recovery: re-attached outpost '{name}' at device #{idx}.")
        missing = [n for n in self._expected_names if n not in present]
        if missing:
            note(f"Auto-recovery: still offline after refresh: {', '.join(missing)}.")

    def _server_busy(self):
        """True while Floxy is speaking — proxied by any outpost being muted (the server
        broadcasts 'stop' to mute every mic for the duration of a reply)."""
        with self._lock:
            return any(o.muted for o in self.outposts)
