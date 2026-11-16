# Linux support

Keep runs on Apple Silicon Macs only. This plan adds Ubuntu and Debian as a second home for the
**server**: chat (Telegram, console, iMessage through a satellite), HTTP, every network integration,
the nightly update and the code-quality run. It is written from a read of the tree on a Mac —
nothing here has been run on Linux yet, and step 1 exists to find what the read missed.

## Scope

**Supported**: Ubuntu 24.04 LTS and newer, Debian 12 and newer, on x86_64 and aarch64, with a
systemd user session. A derivative that names `debian` in `ID_LIKE` (Raspberry Pi OS, Mint) installs
with a warning and is not tested. A machine without a screen is the normal case.

**Decided**:
- **Voice is split: the phones' side runs on Linux, the server's own audio stays on the Mac.**
  A Linux server with `interfaces.voice.enabled` is what the tree already calls a phone-only
  server. *Runs*: everything the iOS app uses in its default mode — TTS and the audio reply
  streamed over `/stream`, `/stream_text`, `/identify_voice`, `/upload_audio_session`, `/log` —
  with the vote, speaker id and the nightly voice discovery behind them. *Mac-only*: the local microphones (MLX Whisper), playback on
  the server's speakers, the ambient-noise volume control (`osascript`), the end / thinking sounds
  and the speakers as a `post_message` destination. No audio device is opened on Linux at all.
  The iOS app is the only remote client: the Android Outposts are no longer supported.
- **The GPU and temperature monitors are hidden** on Linux (`GpuMonitor`, `TempMonitor` and their
  Dashboard cards).

**Mac-only, and said so to the user and the LLM**: the server's own microphones and speakers,
Photos, Reminders, HomeClaw, a *local* Messages host, the iMessage satellite role, the Desktop shortcut, Full Disk Access, the `watchdog` key, the
local-backend start scripts (`scripts/local_ds4_start.sh`, `local_ollama_start.sh`).

**iMessage on Linux** works the way it already does across two Macs: `imessage.host` /
`imessage_chat.host` name a satellite on a Mac. `host: local` on Linux is a configuration error
with a hint, not a crash.

**Not in this plan**: Fedora/Arch, Docker images, Windows/WSL, local audio on Linux
(faster-whisper microphones, PipeWire playback and volume), a desktop launcher, GPU cards from `nvidia-smi`.

## The Mac does not change

Nothing a Mac install does is different after this plan: the same commands run in the same
order, the prompt and the toolkit texts render byte for byte as before, and no config key changes
meaning. Every Linux behaviour is a branch beside the Mac one, never a replacement of it — where
one shared implementation would be simpler (HEIC through `pillow-heif`, video rotation through
`ffmpeg`), the Mac still keeps its own. The only Mac-visible changes are two unused pins leaving
`requirements.txt` (`whisper-openai`, `playsound`). The existing tests are the proof for the code
that is moved rather than added (`start.sh`'s launch, `VoiceInterface`'s imports), and step 5's
real install is repeated on a Mac before the release.

## The one switch: `common/Platform.py`

A new module, the only place that asks the OS what it is. Nothing else tests `sys.platform` after
this plan (the three that do today — `Browser`, `Clipboard`, `WebAccess` — move over; `Telemetry`, which
only reports the OS, is the one name on the allowlist of `tests/unit/common/test_platform.py`).

```python
MAC, LINUX = "mac", "linux"
def current() -> str            # MAC, LINUX, or the raw sys.platform of anything else
def is_mac() -> bool
def sys_platform() -> str       # current() as sys.platform spells it, for requirement markers
def machine() -> str            # "the Mac" / "the server" — the noun user-facing texts use
def mac_only(what: str) -> str  # one wording for every "X needs macOS" answer
```

It imports nothing of the tree (the satellite and `RequirementsChecker` use it before anything else
is loadable). Tests patch `Platform.current`; the unit suite must pass with it patched either way
on either OS — that is what lets the Mac keep testing the Linux branches and CI the Mac ones.

The shell side has the same seam: `scripts/role.sh` sets `os=${KEEP_OS:-$(uname -s)}` and every
script branches on `$os`. `KEEP_OS` exists for the script tests only.

## 1. Dependencies and CI (first, because it finds the unknowns)

- `requirements.txt`: `pyobjc==10.3.2 ; sys_platform == "darwin"`. New, Linux only:
  `pillow-heif` (HEIC, see §6).
- The base file is torch-free: `whisper-openai`, which nothing imported, was the one pin that
  pulled torch in and is gone. Prove it in CI — the base install must not download torch.
- `playsound==1.2.2` has no import anywhere in the tree (the `playsound_*` names are methods of
  `server_audio/PlaySound.py`); the pin is dropped.
- `requirements-voice.txt` is installed on Linux too, minus what only the server's own audio
  needs: `mlx-whisper`, `silero-vad`, `sounddevice` and `lingua-language-detector` get
  `; sys_platform == "darwin"` (each is imported under `local_outposts/` or `server_audio/`
  alone, plus the lazy `sounddevice` of `VoiceInterface.stop`). `soundfile` stays on both: speechbrain
  and supertonic depend on it, `torchaudio.load` (speaker id) can decode through it, and
  `tests/unit/common/test_audio_log.py` reads with it. Its 0.12.1 has no aarch64 wheel — the pure
  one installed there needs the system's libsndfile — so CI, and `ensure_packages` in §3, install
  `libsndfile1`, and `libmagic1` for neonize's python-magic. torch, torchaudio, speechbrain, spaCy
  and the TTS clients stay.
- torch on Linux x86_64 defaults to the multi-GB CUDA build; speaker id does not need it. The
  installer passes PyTorch's CPU wheel index for the voice file, and `RequirementsChecker.install`
  does the same so a nightly pin bump does not swap the build (a `+cpu` local version must
  compare equal to the pin).
- `RequirementsChecker`:
  - `_check_file` evaluates environment markers (`packaging.markers.Marker`, in the venv through
    the pytest pin; imported lazily) and skips a pin whose marker is false. `sys_platform` is taken
    from `Platform.sys_platform()`, so a test reads the files as the other system does. A marker
    that cannot be evaluated — or `packaging` missing — leaves the pin checked: the worst case is
    one pip run, which brings `packaging`.
  - Both groups are checked on both platforms, as today.
  - The Python version check is unchanged (3.12).
- Integration pins to prove by installing, not by reading: `neonize` (WhatsApp, ships a native
  library per platform), `bleak` (BlueZ over D-Bus), `curl_cffi`, `nodriver`, `uvloop`.
- **CI**: a new workflow, `ubuntu-24.04`, x86_64 and arm64 runners: `pip install -r requirements.txt`
  (proving it downloads no torch), then the voice file, then the unit suite with
  `FLOXY_CONFIG=config.example.yaml`. No such job exists today (only
  `pages.yml`). Expect the first run to surface a class the Mac hides: **case-sensitive paths**
  (an import or a `res/` name spelled in another case than the file).

Done when: the unit suite is green on Ubuntu with no test skipped for being on Linux — but for the
tests of Mac-only code that need a library only the Mac installs (`importorskip`: the ten
`TextLanguageDetector` tests on `lingua`, `test_video_rotation`'s three on AVFoundation).

## 2. Process lifecycle: systemd instead of Terminal, applets and launchd

On the Mac the server is launched through Terminal so the Mac's privacy grants apply, started at
login by an applet and revived by a launchd agent. None of that has a reason on Linux. One systemd
**user** unit replaces all of it, and behaves the way a Linux admin expects
(`systemctl --user status keep`, `journalctl --user -u keep`).

**The unit** — `~/.config/systemd/user/keep.service`, written by `scripts/service.sh` (new, bash,
sourced by `start.sh`; it must work before any Python runs):

```ini
[Unit]
Description=Keep
After=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=5
[Service]
WorkingDirectory=<checkout>
ExecStart=<checkout>/.venv/bin/python floxy_ai_server.py
StandardOutput=append:<checkout>/log/raw-floxy.log
StandardError=inherit
Environment=PYTHONUTF8=1
Restart=on-failure
RestartSec=10
[Install]
WantedBy=default.target
```

`ensure_unit` rewrites it when the text differs, runs `daemon-reload` and `enable`. A unit that
already points at another checkout is refused with both paths named — one Keep per user.
`loginctl enable-linger` (allowed for one's own user without sudo) makes it start at boot, before
anyone logs in; the installer runs it once.

`Restart=on-failure` gives Linux for free what the `watchdog` key buys on the Mac, and an explicit
stop stays stopped. A config the process refuses to start on exits non-zero and stops after five
tries instead of looping.

**`start.sh`** keeps its head (role check, `rlog`) and its tail (PR-trial fallback, startup log,
handing the terminal to `tui.py`); the middle — kill the old pids, open Terminal, close the window —
becomes on Linux:

1. Refuse when a process that is not the unit's holds a listen port (a server started by hand).
2. `ensure_unit`, then `systemctl --user restart keep.service`.

One trap: `Updater.relaunch` spawns `start.sh` from inside the server, so the script is in the
unit's cgroup and the restart it asks for would kill it — losing the PR-trial watch. `start.sh`
therefore re-executes itself through `systemd-run --user --scope` when it has no terminal.
`Updater`, `PullRequestTrial` and `restart_server` stay unchanged: they still hand a pid and a
fallback branch to `start.sh`.

**`stop.sh`**: `systemctl --user stop keep.service`, then the existing kill of this checkout's
remaining `floxy_ai_server` processes (the test suite's instances). The watchdog note is Mac-only.

**`autostart.sh`**, `Autostart.py`, `Applet.py`, `DesktopShortcut.py`, `Watchdog.py`: Mac-only.
Each public entry (`Autostart.enable`, `DesktopShortcut.ensure`, `Watchdog.sync`) returns early on
Linux with one log line; `floxy_ai_server.build` and `FirstRun` call them as before.

**`keep.sh`**: with a terminal, as today (attach the console or run `start.sh`). Without one it
exits with a message on Linux — there is no shortcut that would run it that way.
`scripts/cli_start.sh` is Mac-only and untouched.

**`scripts/role.sh`** — the "this user's processes of this checkout" lookups are BSD today:

| | macOS | Linux |
|---|---|---|
| `here` | `/usr/bin/stat -f '0x%Xd %i' .` | `stat -c '%d %i' .` |
| `cwd_of <pid>` | `lsof -p … -d cwd -FDi` | `stat -L -c '%d %i' /proc/<pid>/cwd` |
| `own_listeners` | `lsof -a -u … -iTCP:… -sTCP:LISTEN` | `ss -Hltnp` filtered to the ports, pids owned by `$me` |
| `own_processes` | `pgrep -u … -f` | the same (procps) |

The config reader (awk) and the role logic are portable as they are. A satellite role on Linux is
a `role_problem`: nothing is stopped, nothing started.

**Tests**: `tests/unit/process_table.py` gains `ss`, `stat` and `systemctl` / `systemd-run` /
`loginctl` stubs beside `lsof` and `pgrep`, and its own `stat -f` calls become portable. The
script tests (`test_start_sh`, `test_stop_sh`, `test_role_sh`, `test_keep_sh`, `test_autostart_sh`)
run each branch under `KEEP_OS`, on both OSes. The unit conftest's `Popen` guard refuses
`systemctl`, `systemd-run` and `loginctl`, with a case each in `test_no_server_relaunch.py`.

## 3. `install.sh`

The same one-line install, branching where the Mac is assumed:

- `check_mac` → `check_system`. Linux: `/etc/os-release` (`ID` ubuntu/debian, else `ID_LIKE`
  debian with a warning, else stop), `uname -m` x86_64/aarch64, a working `systemctl --user`
  (stop with the reason otherwise — a container or WSL1), disk space through `df -Pk`, the same
  reachability checks. The terminal advice is skipped.
- `ensure_clt` → on Linux `ensure_packages`: one `sudo apt-get install` of what is missing among
  `git curl ca-certificates ffmpeg iproute2 procps libmagic1 libsndfile1` (§1), plus `python3.12 python3.12-venv` where the
  distribution has them. The command is printed before sudo asks; without sudo the list is shown
  and the install stops.
- `ensure_python`: Ubuntu 24.04 has 3.12; Debian 12 (3.11) and 13 (3.13) do not, so the existing
  **uv** branch is the Linux fallback. `find_python`'s uv glob loses its `macos-aarch64`;
  `python_ok`'s `arm64` assertion becomes "the architecture `check_system` accepted".
- **Chrome** for `WebAccess` (nodriver, already headless, Linux paths already listed): Google
  Chrome's `.deb` on x86_64, the `chromium` package on Debian/arm64. Ubuntu's Chromium is a snap —
  whether nodriver can drive it from a systemd user service is **unverified**; if not, the
  installer says the browse commands need Chrome and goes on.
- `install_requirements`: both files as on the Mac, the voice file with the CPU torch index (§1).
  No PortAudio and no `mpv` from apt: nothing on Linux opens an audio device.
- `KEEP_ROLE=imessage_satellite` on Linux stops: a satellite is a Mac.
- `launch` / `summary`: `enable-linger`, `./start.sh`; the texts lose Terminal, Full Disk Access
  and the microphone prompt on Linux and gain the three `systemctl --user` lines worth knowing.
- Telemetry's chip string comes from `sysctl` — on Linux `uname -m` and the CPU model of
  `/proc/cpuinfo`.
- `tests/unit/test_install_sh.py` covers the Linux branch through `KEEP_OS` and stubbed `apt-get`.

## 4. The server on Linux: what switches off, and how it is said

**Voice.** The interface starts on Linux without its local half. The seams are mostly there
already — `local_outposts.enabled: false` is a supported phone-only setup, `RequestProcessor`
takes `playback_factory=None`, and `_register_room` registers no destination without a local
outpost manager. What changes in `VoiceInterface`, behind one `Platform` check
(`local_audio = Platform.is_mac()`):

- The module-level imports of `server_audio` (`ServerPlayback`, `PlaySound`,
  `DynamicServerVolumeController` — each imports `sounddevice` at load) and of `local_outposts`
  move into the branch that uses them; on Linux none of the three packages is imported.
- `start`: no `start_local_outposts`, no volume controller; `MicrophoneDeps.play_sound` and
  `.volume_controller` become optional, and the routes that use them (`/end`, the settings a
  remote Outpost asks for) skip the sound and answer without the volume.
- `handler_kwargs` leaves `playback_factory` out, so a handler has no `ServerPlayback`: a request
  with `server_sound` (the local outposts, and the app in its server-sound mode) is answered
  as text, logged once.
- `stop` has nothing of the audio teardown to do.
- Unchanged: speaker id, voice discovery, the TTS engines' registration, `/stream_text`, the
  vote. The Outpost TCP server (43523) pushed mute commands to the local microphones and the
  Android Outposts; the iOS app never connects to it, so on Linux it is not started and the port
  leaves `role.sh`'s `listen_ports` there.

Checked against the app (`ViewController.swift`, `TextStreamClient`, `VoiceIdentifier`,
`AudioSessionRecorder`, `PCMStreamPlayer`): by default (`serverSoundMode = false`) it plays the
`/stream` audio itself and calls only the routes listed above. Its server-sound mode — the reply
on the server's speakers, `/end` for the end sound — is the one thing a Linux server cannot do;
there the request is answered as text.

Everything keyed to the server's speakers says so on Linux: `config_set` on `local_outposts.*`
answers `Platform.mac_only(…)`, the VOICES section and the console's Voice tab show no local
microphones (`DashboardTab` imports `LocalOutpostManager` for its mic card — hidden with it), and
`{DESTINATIONS}` lists no room. The local TTS engine (`supertonic`) is to be proven on Linux in
CI or marked Mac-only; ElevenLabs and Google are network clients.

**Integrations.** `Integration` gains a class attribute `mac_only = False`; Photos, Reminders and
HomeClaw set it. On Linux `bootstrap` never constructs such a class, even when
`enabled_integrations` lists it (a config copied from a Mac): the monitor holds a
`DisabledIntegration`-like stand-in whose state reads "needs macOS". `list_integrations` and the
Catalog tab show that, `enable_integration` refuses with the same sentence, and the existing
gates then keep its toolkit, commands and prompt section away from the LLM.

**iMessage.** `Hosts.build` refuses `local` on Linux with a `HealthWarning`-style answer pointing
at `imessage.host` and the satellite install (`SatelliteHints` gets the Linux wording). The
integration and the chat interface are otherwise untouched. `_launch` refuses
`role: imessage_satellite` on Linux through `_refuse_to_start`.

**First run.** `FirstRun`'s shortcut step stamps `desktop_shortcut_at` at once on Linux — no
Full Disk Access probe, no "the grant arrived" turn, no shortcut. The pipeline needs no change:
an install without Photos stays in SETUP and curates nightly, which is already a supported state.

**Setup texts.** The LLM must not send a Linux user to System Settings:
- `show_full_disk_access_setting` joins a new `Commands.MAC_ONLY_COMMANDS`: left out of the
  toolkit's command list on Linux, refused by `Commands.run`.
- `res/toolkits/setup.md` and the first-run brief put their Mac-only prose between
  `<!-- mac -->` / `<!-- /mac -->` lines, dropped on Linux by the same code that handles
  `<!-- keep -->`. "this Mac" elsewhere becomes `Platform.machine()` or a neutral noun.
- `IntegrationsSectionGenerator` drops the Full Disk Access line on Linux and states the platform
  once ("The server runs on Linux; Photos, Reminders, HomeClaw and the server's own microphones and speakers need a Mac").
- `watchdog`: `config_set` answers that systemd already restarts the server.

**Console.** `DashboardTab` builds no `GpuMonitor` / `TempMonitor` on Linux and draws neither
card; the CPU sample (psutil) stays.

## 5. Config and docs

- `config.example.yaml` and the fragments of the Mac-only folders: one "macOS only" line at the
  top of `local_outposts`, `watchdog`, `imessage_satellite`, Photos, Reminders and HomeClaw; the
  voice key under `interfaces` says what a Linux server does with it (the phones, not its own
  microphones and speakers); `imessage` / `imessage_chat` say
  that a Linux server needs `host:` to name a satellite. `config_help` shows them verbatim.
- README: requirements, the install line and what is Mac-only. `CLAUDE.md`: `Platform.py`,
  `scripts/service.sh`, the conftest's new refusals, and the rule "no `sys.platform` outside
  `common/Platform.py`".
- Atlas: `Platform.py` needs its docstring sentence; no new package.

## 6. Small helpers

| Module | Today | Linux |
|---|---|---|
| `common/Browser.py` | `open`, else nothing | `xdg-open` when a display exists; otherwise false, as now. The sign-ins (Google, Dropbox, Volvo, Spotify, Microsoft) are paste-based already and need no browser on the server |
| `common/Clipboard.py` | `pbpaste`, AppleScript | `wl-paste` or `xclip` when installed and a display exists; otherwise empty, as now |
| `assistant/tools/ImageAnalyzer.py` | `sips` HEIC → JPEG | `pillow-heif` — the conversion moves into one function with both ways |
| `common/VideoRotation.py` (Telegram clips) | AVFoundation | `ffmpeg` re-encode, which applies the rotation; on failure the clip goes out as it is, logged |
| `common/SystemTrust.py` | keychain | works as it is (truststore uses the system's OpenSSL store); wording only |
| `common/CacheHitLogMonitor.py`, `WebAccess.py`, `Telemetry.py`, `tui.py` | — | already portable |
| `CodeQualityMaintainer` | `claude`, `gh`, `git` | portable; `gh` and the Claude CLI are the developer's to install, as on the Mac |

`PhotoProcessor`'s `sips` and `DynamicServerVolumeController`'s `osascript` stay: both live behind
a Mac-only switch.

## Things to verify on a real machine

A read cannot settle these; each has a place in the order below.

1. The base requirements install cleanly on x86_64 and aarch64 (§1), neonize above all.
2. Ubuntu's snap Chromium under a user service (§3).
3. `ss -p` shows the pid of the user's own listener without root on both distributions (§2).
4. `systemd-run --user --scope` from inside the unit survives the unit's restart (§2) — the
   PR-trial fallback depends on it.
5. Discovery by multicast (Sonos, HEOS, Hue) with `ufw` on; Plejd over BlueZ needs the `bluez`
   package and the user in the `bluetooth` group. Both become notes in the integrations' config
   comments once seen.
6. A nightly update end to end: pull, pip, restart through the unit, HEAD moved.
7. A voice turn from the iOS app against a Linux server: the spoken reply, the `/stream_text`
   overlay, the speaker's name, a barge-in — and what the app shows when its server-sound mode
   is switched on against a server without speakers.
8. Speaker id's first model load and its speed on CPU-only torch, on x86_64 and aarch64.

## Order of work

Each step leaves `master` releasable for Macs.

1. **Platform.py, requirement markers, RequirementsChecker, CI on Ubuntu** (§1) — until the suite
   is green there. Mac behaviour unchanged.
2. **The switches** (§4) and the helpers (§6), tested under a patched `Platform.current` on the
   Mac — the voice split included: a test imports `VoiceInterface` with `sounddevice` refused and
   starts it as Linux.
3. **Scripts and the unit** (§2), tested through `KEEP_OS` and the staged process table.
4. **install.sh** (§3) and the texts (§5).
5. **A real install** on an Ubuntu 24.04 machine and a Debian 12 one: the verification list, the
   first-run conversation from an empty config, a restart, a nightly update, a PR trial.
6. Release; the readme names Linux only after step 5.
