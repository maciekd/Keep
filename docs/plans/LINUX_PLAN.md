# Linux support

Keep runs on Apple Silicon Macs only. This plan adds Ubuntu and Debian as a second home for the
**server**: chat (Telegram, console, iMessage through a satellite), HTTP, every network integration,
the nightly update and the code-quality run. Steps 1–4 of the order of work are built; they were
written and tested on a Mac, with the unit suite also run on Ubuntu in CI. The checklist at the end
has been worked through once on Ubuntu 24.04 (aarch64, a VM): each line says what was seen. Debian,
x86_64, a real phone and the Mac's own lines are what still stands between this and a release.

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
local-backend start scripts (`scripts/local_ds4_start.sh`, `local_omlx_start.sh`).

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
`requirements.txt` (`whisper-openai`, `playsound`) and the installer's temp file for a log outside
the install folder, which `mktemp` now makes from a full template (§3). The existing tests are the proof for the code
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
StartLimitIntervalSec=300
StartLimitBurst=5
[Service]
WorkingDirectory=<checkout>
ExecStart=<checkout>/.venv/bin/python floxy_ai_server.py
StandardOutput=append:<checkout>/log/raw-floxy.log
StandardError=inherit
Environment=PYTHONUTF8=1
Environment="PATH=%h/.local/bin:%h/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin"
Restart=on-failure
RestartSec=10
[Install]
WantedBy=default.target
```

The `PATH` line is there because a unit gets the user manager's `PATH`, which has no
`~/.local/bin` — where the Claude CLI of the nightly code-quality run, `uv` and whatever else the
user installed live. A unit cannot add to that `PATH`, only replace it, so the line names all of
it: the user's two folders first, systemd's default folders, and `/snap/bin`, which Ubuntu adds
for a login and where its Chromium is.

`ensure_unit` rewrites it when the text differs and then runs `daemon-reload`; `enable` runs at
every start. A unit that already points at another checkout is refused with both paths named —
one Keep per user (told by device and inode, as `role.sh` tells every folder; a unit whose folder
is gone is taken over, a file of that name without a `WorkingDirectory` is left alone). The path
goes into the unit as it is, so a checkout whose path has a space, a quote, a backslash, `$` or
`%` is refused: each means something else in `ExecStart` than in the two path settings, and no
quoting of it could be tried from a Mac.
`loginctl enable-linger` (allowed for one's own user without sudo) makes it start at boot, before
anyone logs in; the installer runs it once.

The unit orders itself after nothing. `After=network-online.target` stood here and ordered
nothing: the target is the system manager's, and a user manager does not know it (`not-found` on
Ubuntu 24.04 and on Debian 12). The server starts as soon as the user manager does — on the
machines tried, a fraction of a second after the network was online — and what it reaches over the
network it retries; whether a slower network is ever a problem has not been seen. A unit written
with the line is rewritten without it at the next start (`ensure_unit` compares the text).

`Restart=on-failure` gives Linux for free what the `watchdog` key buys on the Mac, and an explicit
stop stays stopped. A config the process refuses to start on exits non-zero and stops after five
tries instead of looping.

**`start.sh`** keeps its head (role check, `rlog`) and its tail (PR-trial fallback, startup log,
handing the terminal to `tui.py`); the middle — kill the old pids, open Terminal, close the window —
becomes on Linux:

1. Refuse when a process that is not the unit's holds a listen port (a server started by hand).
   "The unit's" is asked of `/proc/<pid>/cgroup`, not of the unit's main pid. The exception is the
   pid handed over when it is that listener: a server started by hand that asks for its own
   restart (`restart_server`) is stopped here and the unit runs the new one — refusing it left the
   command answered with nothing.
2. `ensure_unit`, `systemctl --user reset-failed` (a unit that used up its start limit refuses a
   start by hand too), then `systemctl --user restart keep.service`.

Each system's launch is a file of its own, sourced by `start.sh` between its head and its tail:
`scripts/launch_mac.sh` is the Mac's middle moved as it was, `scripts/launch_linux.sh` the steps
above. It also refuses to go on without `ss` — with no listener ever seen, every start would pass
for one that never came up.

One trap: `Updater.relaunch` spawns `start.sh` from inside the server, so the script is in the
unit's cgroup and the restart it asks for would kill it — losing the PR-trial watch. `start.sh`
therefore runs itself again through `systemd-run --user --scope` when it finds itself in the
unit's cgroup (`/proc/self/cgroup` — the condition itself, where "no terminal" was a guess at
it). It asks for a scope with `true` first, so a systemd that gives none costs only the watch and
not the restart; the copy left in the unit waits for the scoped one and ends with the unit, and
`KEEP_START_SCOPED` keeps it from trying twice. The pid `Updater` hands over is not killed on
Linux: the restart stops it. `Updater`, `PullRequestTrial` and `restart_server` stay unchanged:
they still hand a pid and a fallback branch to `start.sh`.

**`stop.sh`**: `systemctl --user stop keep.service` — when the unit is this checkout's — then the
existing kill of this checkout's remaining `floxy_ai_server` processes (the test suite's
instances, a server started by hand). The unit stays enabled. The watchdog note is Mac-only.

**`autostart.sh`**, `Autostart.py`, `Applet.py`, `DesktopShortcut.py`, `Watchdog.py`: Mac-only.
Each public entry (`Autostart.enable`, `DesktopShortcut.ensure`, `Watchdog.sync`) returns early on
Linux with one log line; `floxy_ai_server.build` and `FirstRun` call them as before.

**`keep.sh`**: with a terminal, as today (attach the console or run `start.sh`). Without one it
exits with a message on Linux — there is no shortcut that would run it that way.
`autostart.sh` and `scripts/cli_start.sh` are Mac-only: on Linux each says so and exits non-zero.

**`scripts/role.sh`** — the "this user's processes of this checkout" lookups are BSD today:

| | macOS | Linux |
|---|---|---|
| `here` | `/usr/bin/stat -f '0x%Xd %i' .` | `stat -c '%d %i' .` |
| `cwd_of <pid>` | `lsof -p … -d cwd -FDi` | `stat -L -c '%d %i' /proc/<pid>/cwd` |
| `own_listeners` | `lsof -a -u … -iTCP:… -sTCP:LISTEN` | `ss -Hltnp` filtered to the ports, pids owned by `$me` (`stat -c %u /proc/<pid>`) |
| `own_processes` | `pgrep -u … -f` | the same (procps) |

The config reader (awk) and the role logic are portable as they are. A satellite role on Linux is
a `role_problem`: nothing is stopped, nothing started.

**Tests**: `tests/unit/process_table.py` gains `ss`, `stat` and `systemctl` / `systemd-run` /
`loginctl` stubs beside `lsof` and `pgrep`. The Mac branch asks `/usr/bin/stat` by its path, so
where that file is GNU's the table brings a bash function of that name answering in BSD's words.
The script tests (`test_start_sh`, `test_stop_sh`, `test_role_sh`, `test_keep_sh`, `test_autostart_sh`)
run each branch under `KEEP_OS`, on both OSes — but for `test_role_sh`'s own-process cases, which
ask the real tools of the system the suite is on (on Ubuntu: `pgrep` and `/proc`). The unit conftest's `Popen` guard refuses
`systemctl`, `systemd-run` and `loginctl`, with a case each in `test_no_server_relaunch.py`.

## 3. `install.sh`

The same one-line install. Every Linux step stands beside the Mac's behind `linux` (the script's
`OS`, `KEEP_OS` for the tests); `check_mac`, `ensure_clt` and the Mac's summary and launch are
untouched.

- **Refused before a folder or a log is made**: root, `KEEP_ROLE=imessage_satellite` (a satellite is a
  Mac — and an existing config.yaml that names the role, after it is read), and a `KEEP_DIR` whose
  path `start.sh` would refuse for the unit (a space, a quote, a backslash, `$`, `%`, a control
  character).
- `check_linux`: `/etc/os-release` read with `sed`, never sourced (`ID` ubuntu/debian, else
  `ID_LIKE` naming debian or ubuntu with a warning, else stop; a release older than 24.04 / 12 is
  warned about, not refused), `uname -m` x86_64/aarch64, `systemctl --user show-environment` as
  the proof of a user session (stop with systemd's own words otherwise — a container, WSL without
  systemd, a shell reached through `su`), disk space through `df -Pk`, the same reachability
  checks. The terminal advice is skipped. A machine without `curl` (the script came
  through `wget`) has its network tested after the apt step instead.
- `ensure_packages` where the Mac has the Command Line Tools: one
  `sudo apt-get install -y --no-install-recommends` of what `dpkg-query` does not have among
  `git curl ca-certificates ffmpeg iproute2 procps libsndfile1 libmagic1 openssh-client` (§1; the
  last for `ssh-keygen`, which git verifies a release's signed tag with) — a library counts
  under its `t64` name too, which is what Ubuntu 24.04 calls `libmagic1` — plus `python3.12` and
  `python3.12-venv` where no usable 3.12 was found and `apt-cache policy` has a candidate. The
  command is printed before sudo asks and runs with its stdin taken away. Without sudo, or with a
  sudo that needs a password and no terminal, the command is shown and the install stops.
  **`apt-get update` is run only when it is needed**: an install that fails with apt's "Unable to
  locate package" — empty or old lists, which is how a fresh cloud image of either distribution
  comes (checklist, line 2) — is answered with `sudo apt-get update`, shown first, and one more
  try of the install (the missing list taken again, since apt may only now offer a `python3.12`).
  An install that works the first time updates nothing, and a second failure stops with apt's own
  words; one that failed for another reason keeps the hint to update the lists and re-run.
- `ensure_python`: the system's 3.12 when it passes `python_ok` — which on Linux asks for
  `ensurepip` as well, since Debian and Ubuntu ship it apart and a `python3.12` without it imports
  `venv` and then fails to make one — else **uv**, never Homebrew. `find_python`'s uv glob names
  the build for this machine (`linux-<arch>-gnu`); `python_ok` asserts `uname -m` on Linux where
  the Mac asserts arm64.
- **Chrome** for `WebAccess`: the installer looks for the four names `WebAccess` looks for (on
  PATH and in `/snap/bin`, which the unit's PATH has and an ssh command's may not) and, without
  one, asks whether to install it (`ensure_browser`, the step after the system packages, so sudo
  still knows its password; `KEEP_BROWSER=1|0` answers for it, and without a terminal nothing is
  installed) — Google Chrome's `.deb` on x86_64, the `chromium` package on Debian, the `chromium`
  snap on Ubuntu/ARM. A no or a failed install does not stop the install: the summary says what
  to run. *The installing itself is not yet proven on a real machine — only the commands were,
  typed by hand (checklist, line 22); Chrome's `.deb` on x86_64 not at all.*
- `install_requirements`: both files as on the Mac, the voice file with the CPU torch index (§1).
  No PortAudio and no `mpv` from apt: nothing on Linux opens an audio device.
- `enable_linger` (`loginctl --no-ask-password enable-linger <user>`) before the summary; a
  refusal is a warning in the summary with the sudo command, since without lingering the server
  stops when the installing session ends.
- `summary` / `launch`: no Terminal, no Full Disk Access, no microphone prompt. The summary names
  `systemctl --user status keep`, `journalctl --user -u keep`, `log/raw-floxy.log`,
  `systemctl --user disable keep.service` (what `stop.sh` leaves on), and `ssh -t … keep.sh` as
  the way to the console and the setup conversation on a machine without a screen. A start needs
  no terminal on Linux, so an install over plain ssh (`ANTHROPIC_API_KEY` in the environment) ends
  with `start.sh` run and the unit up, where the Mac says "start Keep with: start.sh".
- Telemetry: `os=Linux`, the distribution's name and `uname -m` with the CPU model of
  `/proc/cpuinfo` as the chip; the install id falls back to the kernel's uuid where there is no
  `uuidgen` (Debian has it in `uuid-runtime`).
- BSD and GNU: `mktemp -t <prefix>` became a full template (`$TMPDIR/keep-install.XXXXXX`), which
  both take — the one line that changed for the Mac too; `df` and `mktemp` are called by their
  absolute path where it exists; `sysctl`, `sw_vers`, `xcode-select` and `scutil` stay inside
  the Mac's functions.
- `tests/unit/test_install_sh.py` covers the Linux branch through `KEEP_OS`, with `sudo`,
  `apt-get`, `dpkg-query`, `apt-cache`, `systemctl` and `loginctl` as functions of the harness.

## 4. The server on Linux: what switches off, and how it is said

**Voice.** The interface starts on Linux without its local half. The seams are mostly there
already — `local_outposts.enabled: false` is a supported phone-only setup, `RequestProcessor`
takes `playback_factory=None`, and `_register_room` registers no destination without a local
outpost manager. What changes in `VoiceInterface`, behind one `Platform` check
(`local_audio = Platform.is_mac()`):

- The module-level imports of `server_audio` (`ServerPlayback`, `PlaySound`,
  `DynamicServerVolumeController` — each imports `sounddevice` at load) and of `local_outposts`
  move into the branch that uses them; on Linux nothing of `server_audio` is imported, and of
  `local_outposts` only `constants` and `scribe_status`, which the reply-timing gate
  (`VoiceTurnGate`) reads on every voice server — the package's `__init__` no longer imports the
  outposts' machinery.
- `start`: no `start_local_outposts`, no volume controller; `MicrophoneDeps.play_sound` and
  `.volume_controller` become optional, and the routes that use them (`/end`, the settings a
  remote Outpost asks for) skip the sound and answer without the volume.
- `handler_kwargs` leaves `playback_factory` out, so a handler has no `ServerPlayback`: a request
  with `server_sound` (the local outposts, and the app in its server-sound mode) is answered
  as text, logged once — in the `/stream` route (`HttpDeps.server_sound`), beside the existing
  downgrade of an audio client on a voice-less server, and whether voice is enabled or not.
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
`MacOnlyIntegration`, a `DisabledIntegration` whose state is DISABLED — so every existing gate
reads it as off without knowing about it — with the reason as its error and `needs_mac` in its
status. `list_integrations` and the Catalog tab show "needs macOS", `enable_integration` refuses
with the same sentence (`disable_integration` still takes it off the list), the contracts' "enable
it" hints leave it out, and the gates keep its toolkit, commands and prompt section away from the LLM.

**iMessage.** `Hosts.build` answers `local` on Linux with a host that sends nothing and reports
NOT_CONFIGURED (`RemoteMessagesHost.unusable`), so both consumers show it as the health problem
they already have a path for; `SatelliteHints.problem` words it for Linux, pointing at
`<section>.host` and the satellite install. The integration and the chat interface are untouched. `_launch` refuses
`role: imessage_satellite` on Linux through `_refuse_to_start`.

**First run.** `FirstRun`'s shortcut step stamps `desktop_shortcut_at` at once on Linux — no
Full Disk Access probe, no "the grant arrived" turn, no shortcut. The pipeline needed one change:
its SETUP stage read `enabled_integrations` for Photos, so a config copied from a Mac would have
sent it to INDEXING to wait for a feed that is never built. On Linux it stays in SETUP and curates
nightly, which is already a supported state.

**Setup texts.** The LLM must not send a Linux user to System Settings:
- `show_full_disk_access_setting` joins a new `Commands.MAC_ONLY_COMMANDS`
  (`Toolkits.MAC_ONLY_COMMANDS`): left out of `Toolkit.offered_commands()` on Linux, refused by
  `Commands.run`.
- `res/toolkits/setup.md` and the first-run brief put their Mac-only prose between
  `<!-- mac -->` / `<!-- /mac -->` lines, dropped on Linux by the same code that handles
  `<!-- keep -->` (`Rewriter.gate_platform`). The markers are whole lines, so where a sentence
  must read differently — a list of commands, "this Mac" in the middle of a paragraph — the Linux
  wording stands beside it between `<!-- linux -->` / `<!-- /linux -->` lines, dropped on a Mac.
  That keeps the Mac's rendering byte for byte; `Platform.machine()` could not, since the texts
  say "this Mac". The iCloud toolkit's 2FA paragraph is written the same way; the integration's
  own hints, `SatelliteHints` and the watchdog key's hint pick their wording by `Platform`.
- `IntegrationsSectionGenerator` drops the Full Disk Access line on Linux and states the platform
  once ("The server runs on Linux; Photos, Reminders, HomeClaw and the server's own microphones and speakers need a Mac").
- `watchdog`: `config_set` answers that systemd already restarts the server.

**Console.** `DashboardTab` builds no `GpuMonitor` / `TempMonitor` on Linux and draws neither
card; the CPU sample (psutil) stays. The tiles that are left share the row (two summary tiles,
one usage chart, two bottom charts), the mic noise floor tile goes with the microphones, and the
resource rows lose the three that read the Mac (processed photos, volume, background noise).

## 5. Config and docs

- `config.example.yaml` and the fragments of the Mac-only folders: one "macOS only" line at the
  top of `local_outposts`, `watchdog`, `imessage_satellite`, Photos (in the root, above the
  comment that already stands over its pointer), Reminders and HomeClaw; the voice key under
  `interfaces` says what a Linux server does with it (the phones, not its own microphones and
  speakers); `imessage` says that a Linux server needs `host:` to name a satellite and carries
  the satellite's install command, `imessage_chat` that the account then always lives on a Mac
  with a satellite. `config_help` shows them verbatim, and the Linux first-run brief and
  `setup.md` send the assistant to `config_help("imessage")`.
- README: Linux named as new and in testing, with what stays on the Mac. `CLAUDE.md`: the Linux
  install under Releases, the CPU torch index and the apt packages.
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

The code was written from documentation, on a Mac. Work
through it in this order on **an Ubuntu 24.04 machine** (U), **a Debian 12 machine** (D) — one of
the two x86_64 and one aarch64 if possible — and, last, **a Mac** (M). A line without a letter is
for both Linux machines. Each failed line is a bug to fix before the release, or a sentence to
correct in the installer, the config comments or this plan.

**Seen so far** — the *U:* note under a line: Ubuntu 24.04.5, aarch64, a Lima VM (4 cores, 6 GB,
no audio, no Bluetooth, no LAN of its own), 2026-11-16 — first without an Anthropic API key, on a
placeholder; the model's side was seen in a second pass with a real key. The *D:* note: Debian 12
(bookworm, systemd 252, ffmpeg 5.1, Python 3.11 only), aarch64, the same kind of VM, the same day —
on a placeholder key but for one install with a real key and two turns of the model. *Proven* is
what was watched to work there; nothing has been seen on x86_64 or on the Mac yet. In both VMs the
user's uid is 501 (Lima's), below the 1000 from which journald keeps a journal per user, so the
unit's lines are in the system's journal: `journalctl --user -u keep`, which the summary names,
finds no journal there on Debian (systemd 252) and shows the unit on Ubuntu (255), and `journalctl
--user-unit keep` shows it on both. A VM's artefact: an account made with uid 1000 is shown its
unit by both forms on both systems, so the summary keeps the first (the second adds a notice about
the system's messages for a user outside `adm`).
After the Debian run's fixes Ubuntu was seen once more, with the real key: config.yaml and the new
files of `cache/config-backups/` stay 600 through a `config_set` (the model's, and the command's
code under umask 0002); asked how the server starts at boot and how to open the console from
another computer, the model answered from the setup toolkit (keep.service, `ssh -t <user>@<host>
~/Keep/keep.sh`, no `lan_access`); the unit suite passes in the install's venv (13701 passed, 55
skipped) and the logs hold no traceback, error or warning.
The unit suite was run in the Debian install's own venv as well: one test failed there every time —
two writes of a file a few milliseconds apart carry the same mtime on that kernel, and
`VoiceCandidates.version` told them apart by mtime; it now tells by content — and the rest passed.

**A. The install**

1. Fresh machine, logged in over ssh as a normal user with sudo: the readme's line. `check_linux`
   names the distribution and the CPU, finds the user session (`systemctl --user
   show-environment`), and the network.
   *U: proven, with a terminal and without.*
2. The apt step lists only what is missing, shows the command, and sudo's password prompt is
   readable under it; apt's output is in `log/install.log`. U: `libmagic1` installs as
   `libmagic1t64` and a re-run asks for no sudo. D: stale apt lists — does the install fail with
   the "apt-get update" hint, and is that hint enough?
   *U: proven — and the stale lists are Ubuntu's too: on a fresh cloud image apt knew no `ffmpeg`
   ("Unable to locate package"), the install stopped with the hint, and `sudo apt-get update` plus a
   re-run was enough. sudo was passwordless, so its prompt was not seen. Open: whether the installer
   should run the update itself, since a new VM's first install fails this way.*
   *D: the same — Debian's cloud image comes with no lists at all (Lima's own provisioning had
   fetched them; emptied again for the run): "Unable to locate package git", and the hint was
   enough. Decided since: the installer now runs `sudo apt-get update` itself when apt knows a
   package by no name, says so, and tries once more (§3) — proven on Debian from empty lists;
   Ubuntu to be seen again. Missing on the image were `git ffmpeg libsndfile1` alone; `libmagic1`
   is `libmagic1` there and a re-run asks for no sudo.*
   *U, after it: proven — lists emptied and `ffmpeg` removed, a re-run over the install said so, ran
   `sudo apt-get update` and the install a second time and went on to its end, config.yaml
   untouched; with lists and nothing missing it calls sudo not once (a logging `sudo` in front).*
3. Python. U: `python3.12` + `python3.12-venv` from apt, `PY_SOURCE=found`. D: no 3.12 in apt, uv
   installs one, `find_python`'s glob finds it on a re-run (`cpython-3.12.*-linux-<arch>-gnu`),
   and `truststore` / `SystemTrust` verify TLS with uv's Python (its OpenSSL looks in /etc/ssl).
   *U: proven (`python3.12-venv` alone was missing). With apt's lists empty there is no candidate
   and uv would be taken instead — not seen, the apt step stops first.*
   *D: proven. No candidate in apt, uv 0.13.8 installed to `~/.local/bin` (no profile touched),
   Python 3.12.16 in a second. uv names the folder `cpython-3.12.16-linux-aarch64-gnu` and puts a
   `cpython-3.12-linux-aarch64-gnu` link beside it, which is what `uv python find` answers and the
   venv is made from; the glob finds the first on a re-run with `~/.local/bin` off the PATH, and
   with it on (Debian's `.profile` adds the folder once it exists) `python3.12` there is found
   first. TLS verifies with uv's Python (OpenSSL 3.5, `/etc/ssl/certs`) through `truststore`:
   Anthropic, GitHub, PyPI, Telegram and Hugging Face answered, a self-signed host was refused.*
4. `pip install` with the CPU index: no `nvidia-*` wheel in the log, `torch.version.cuda is None`;
   every pin installs on both architectures — `neonize` above all, `bleak`, `curl_cffi`,
   `nodriver`, `uvloop`, `pillow-heif`, and `soundfile` against the system's libsndfile on aarch64.
   The progress bar is scaled for the Mac's package count (`PIP_EXPECTED_PACKAGES`): note how far
   off it is.
   *U: proven on aarch64 — no `nvidia-*` wheel, `torch.version.cuda is None`, every pin installs
   and imports (neonize with libmagic, soundfile on libsndfile 1.2.2), 2½ minutes, a 1.6 GB venv,
   3 GB of disk with the apt packages. 228 packages against the Mac's 380: the bar stopped at 60 %,
   fixed with `PIP_EXPECTED_PACKAGES_LINUX`. aarch64's torch wheel carries no `+cpu`, so the
   comparison of a `+cpu` version with its pin is x86_64's to show.*
   *D: proven on aarch64 with uv's Python — the same 228 packages in 1m 45s, no `nvidia-*` wheel
   (the one "nvidia" in the log is apt's list of suggestions), `torch.version.cuda is None`,
   everything imports (soundfile on libsndfile 1.2.0, libheif 1.19.7 from the wheel), 1.6 GB.*
5. `loginctl enable-linger` succeeds without sudo and without a polkit prompt over ssh
   (`loginctl show-user $USER -p Linger`). If it is refused, the summary's warning shows.
   *U: proven (turned off first; on again without sudo or a prompt). The refusal was not seen.*
   *D: proven the same way, from an install without a terminal.*
6. The summary reads right on a real terminal, and its `ssh -t user@host ~/Keep/keep.sh` line
   works from another machine with the printed host name.
   *U: proven as far as a VM goes: `ssh -t … keep.sh` from the Mac opens the console; the printed
   host name is the VM's own, which nothing outside it resolves.*
   *D: the same; the console opened over `ssh -t` from the Mac. The summary now also says that
   the server answers this machine only until `interfaces.lan_access` is on.*
7. The refusals, each on a throwaway run: as root; `KEEP_ROLE=imessage_satellite`;
   `KEEP_DIR="$HOME/My Keep"`; inside a Docker container (no systemd); through `sudo -u other -i`
   (no user session).
   *U: proven but for Docker, which was not tried. Root was refused only after its log folder was
   made (`/root/Keep/log`) — fixed, it is refused before. `sudo -u other -i` is refused with
   systemd's "Failed to connect to bus: No medium found".*
   *D: the same four proven (root before any folder is made), Docker not tried either.*
8. `ssh host 'curl … | ANTHROPIC_API_KEY=… bash'` (no terminal): the install ends with the unit
   running. With `NOPASSWD` sudo it gets through the apt step; without, it stops with the command.
   *U: the `NOPASSWD` half proven, and without a key it stops at the key step with "No terminal to
   ask on". With a real key in the environment: not done (none at hand) — the run to its end was a
   re-run over a config.yaml written by hand. A sudo that needs a password: not seen.*
   *U, second pass: the success path proven with a real key and no terminal — all eight steps, the
   key verified, config.yaml 600, the unit running — and at the prompt in tmux (hidden typing, a
   rejected key asked again, a good one verified, the console opened). A key Anthropic rejects
   exits 1 and leaves no config.yaml.*
   *D: the same, each seen: no key and no terminal stops at the key step; a rejected key exits 1
   with no config.yaml; a real key in the environment and no terminal runs all eight steps on a
   fresh folder (key verified, config.yaml 600, the key in no log, the unit running). The prompt
   itself was not typed at. A sudo that needs a password: not seen.*
9. A re-run on the finished install: every step is skipped or fast, config.yaml kept, the server
   restarted.
   *U: proven (4 s).*
   *D: proven (4 s).*

**B. The unit and the scripts**

10. `systemctl --user status keep` shows the unit active; `log/raw-floxy.log` receives the
    output (`StandardOutput=append:`); `systemctl --user show keep -p Environment` has the PATH
    with the home folder expanded from `%h`, and `which claude` from inside the server (a
    `self_analysis` run) finds `~/.local/bin/claude`.
    *U: proven but for the Claude CLI, which was not installed: the PATH is in the server's
    environment with the home folder expanded.*
    *D: the same, on systemd 252 (`append:` and `%h` work there); the Claude CLI was not installed.*
11. Reboot with nobody logged in: the server is up (lingering, `WantedBy=default.target`; the
    unit waits for no network target, §2).
    *U: proven — the unit started 0.3 s after the user manager and before the first login.
    `network-online.target` does nothing there (§2).*
    *D: proven — the unit started with the user manager (0.2 ms after it, 0.1 s after the
    system's network-online.target, a second before the first login), written without the
    `After=` line, which is gone from the unit (§2); an install's old unit was rewritten at its
    next start.*
    *U, again: the `//restart` that pulled the change rewrote the install's old unit without the
    line; after a stop and a start of the VM the unit was started 0.7 s before the first login.*
12. `ss -Hltnp` shows the pid of the user's own listener without root, in the format
    `own_listeners` parses (`users:(("python",pid=…`), with IPv4 and IPv6 sockets.
    *U: proven, a `*:port` dual-stack listener included; root's sshd shows no pid and is not ours.*
    *D: proven (iproute2 6.1): the same format; with `lan_access` on the listeners are
    `0.0.0.0:8888` and `0.0.0.0:5151`.*
13. `./stop.sh` stops the unit and it stays down; `./start.sh` brings it back; `kill -9` of the
    server is followed by a restart after 10 s (`Restart=on-failure`).
    *U: proven (answering again 12 s after the kill).*
    *D: proven (12 s).*
14. A config the server refuses (delete `vault.key`): five starts, then the unit is failed —
    and `./start.sh` after fixing it starts again (`reset-failed` clears the start limit).
    *U: proven. The limit counts every start of the last five minutes, a start by hand too: the
    unit was failed after three tries here.*
    *D: proven — failed at the fifth restart, 55 s after the start; `./start.sh` after the fix
    started it.*
15. `restart_server` from the chat: `start.sh` finds itself in the unit's cgroup
    (`/proc/self/cgroup`), `systemd-run --user --scope` takes it out, the restart happens, and
    the copy left in the unit ends on the stop's SIGTERM without holding it up.
    *U: proven with `//restart`, which runs `restart_server`'s code (the stop took 2–3 s).*
    *U, second pass: `restart_server` through the model, approved in the console — back in 2 s,
    the console reconnected, the history kept.*
    *D: proven with `//restart` on systemd 252 (2 s) — and on an image without
    `dbus-user-session`, so with no user bus at all: `systemd-run --user --scope` and every
    `systemctl --user` of the scripts work over the manager's private socket.*
16. A server started by hand (`.venv/bin/python floxy_ai_server.py` in a terminal, unit
    stopped): `./start.sh` refuses, naming the pid; `restart_server` from that server's chat —
    what does the user see? `./stop.sh` stops it.
    *U: `start.sh` refuses and `stop.sh` stops it — proven. `restart_server` there: the user saw
    "restarting" and nothing happened, the start refused its own caller. Fixed (§2): that server is
    stopped and the unit takes over.*
    *D: all three proven, with the fix in.*
17. `./keep.sh` with a terminal attaches the console; without one (`ssh host ~/Keep/keep.sh`) it
    says so; `autostart.sh` and `scripts/cli_start.sh` refuse.
    *U: proven.*
    *D: proven (the console over `ssh -t` from the Mac).*
18. A checkout path with a letter outside ASCII (`~/Kęp`) starts.
    *U: proven — started, restarted from the chat and stopped there (the install moved to it, so
    the unit of a folder that is gone was taken over as well).*
    *D: proven the same way, and moved back.*

**C. The server**

19. The first-run conversation from an empty config, in the console over `ssh -t`: the Linux
    brief (no System Settings, no Photos), Telegram set up, the pipeline staying in SETUP.
    *U: not done — no API key. The Linux brief is what the prompt carried and the pipeline logged
    that it stays in SETUP; the conversation itself has not been seen.*
    *U, second pass: proven in the console the installer opens, except Telegram — the Linux brief
    with no Mac advice, the owner's name saved, voice for the phone app set up by the assistant,
    the pipeline staying in SETUP.*
    *D: not done as a conversation. One turn of the real model, over HTTP: it named the three
    integrations that need a Mac, the satellite for iMessage and the phone app for voice, and gave
    `ssh -t <user>@<host> ~/Keep/keep.sh` for the console, saying that no `lan_access` is needed.*
20. `list_integrations`: Photos, Reminders and HomeClaw read "needs macOS"; `enable_integration`
    refuses them; `config_set("watchdog", …)` and `local_outposts.*` answer as §4 says.
    *U: proven by calling the commands in the install's venv (no model to call them) and in the
    Catalog tab.*
    *U, second pass, through the model: the three read as needing a Mac, Photos refused,
    `watchdog` and `local_outposts.enabled` refused, `show_full_disk_access_setting` not offered.*
    *D: spot-checked — the three are "needs macOS — not built" in the unit's log and in the model's
    answer; `config_set` refuses `watchdog` and `local_outposts.enabled` (called in the venv).*
21. The console's Dashboard without the GPU, temperature and microphone cards.
    *U: proven.*
    *D: not looked at — only the log line that the tiles are left out.*
22. Browsing. D: `chromium` from apt, driven headless by nodriver from the unit. U x86_64: Google
    Chrome's `.deb`. U: the snap Chromium (`/snap/bin` is on the unit's PATH) — does nodriver
    start it from a systemd user service? If not, say so in the installer's advice.
    *U (aarch64): proven — the snap Chromium starts under nodriver from the unit and from a
    `systemd-run --user` service: a page opened, snapshot and screenshot taken. The installer's
    advice now names the snap. Chrome's `.deb` on x86_64: not tried.*
    *U, second pass: through the model, example.com in 2 s.*
    *D: proven — `sudo apt-get install -y chromium` (156, the installer's advice), found at the
    next start with no reinstall; started by nodriver in the unit and from a `systemd-run --user`
    service (a page opened, snapshot, screenshot, a fetch), and through the model (example.com's
    title, 6 s with the toolkit load). The package brings `dbus-user-session` along.*
23. A HEIC photo sent over Telegram is described (`pillow-heif`); a rotated phone clip arrives
    upright (`VideoRotation._bake_with_ffmpeg`: the `-vf` scale filter, the codec flags, the
    rotation applied and no display matrix left); a Telegram voice message is transcribed.
    *U: the two conversions proven on files made for it (a HEIC → JPEG; a 1920x1080 clip with a
    90° matrix → 720x1280 with none, 0.3 s), not through Telegram. The voice message: not tried
    (Google's STT key).*
    *D: the two conversions proven the same way, on ffmpeg 5.1: a HEIC → JPEG; a 1920x1080 clip
    with a 90° matrix → 720x1280 with none, 0.3 s (ffmpeg 5.1 has no `-display_rotation` to make
    such a clip with, so the matrix was written into the file by hand). Not through Telegram.*
24. On a desktop session only: `Clipboard` through `wl-paste --no-newline --type …` (Wayland) and
    `xclip -selection clipboard -o -t …` (X11); `Browser` through `xdg-open`. Headless: both do
    nothing, quietly.
    *U: headless proven. A desktop session: not tried.*
    *D: headless proven. A desktop session: not tried.*
25. Discovery by multicast (Sonos, HEOS, Hue) with `ufw` on; Plejd over BlueZ (the `bluez`
    package, the user in the `bluetooth` group). What is found becomes a note in the
    integrations' config comments.
    *U: not possible in the VM. `bleak` without BlueZ answers "org.bluez was not provided".*
    *D: not possible in the VM; the same answer from `bleak`.*
26. WhatsApp links (neonize's native library, `libmagic1`).
    *U: the library loads; linking an account was not tried.*
    *D: the same.*
27. iMessage: `host: local` shows as the configuration problem with the Linux wording; with a
    satellite on a Mac (`lan_access: true`) the feed reads, sends, and a satellite that is down
    gives the hints of `SatelliteHints` without "this Mac". An iCloud sign-in's 2FA hint likewise.
    *U: `host: local` proven (not configured, "Messages runs on a Mac only"). A satellite: not tried.*
    *D: `host: local` proven the same; a satellite: not tried.*

**D. Voice**

28. `interfaces.voice.enabled: true`: the server starts without opening an audio device, speaker
    id loads its model on CPU-only torch (time the first load and one identification, x86_64 and
    aarch64), `supertonic` synthesizes — or is to be marked Mac-only.
    *U (aarch64): proven. The speaker model loads in 5 s the first time (with its download), an
    embedding of a 3 s clip takes 0.07–0.1 s; `supertonic` loads in 0.8 s, warms up in 1.7 s and
    synthesizes 3.8 s of speech in 0.9 s — it stays on both systems. A phone-style `/stream`
    request was answered with audio.*
    *U, second pass, with the real model: a phone-style `/stream` with the sentence on
    `/stream_text` and 2.75 s of Supertonic audio, the first chunk after 0.6 s.*
    *D (aarch64): proven on a placeholder key. Voice on starts no audio device; the speaker model
    loads in 5.7 s the first time (with its download; 1.2 s after), an embedding of a 3 s clip
    takes 0.13–0.18 s; `supertonic` loads and warms up in 7 s with its download (1.4 s after, 2.3 s
    in the server) and synthesizes 7.7 s of speech in 4 s. A phone-style `/stream` was answered
    with audio.*
29. The iOS app against the Linux server: the spoken reply, the `/stream_text` overlay, the
    speaker's name, a barge-in — and what the app shows with its server-sound mode on against a
    server without speakers (answered as text).
    *U: not done (no phone). With curl: a `server_sound` request is answered as text and logged.*
    *U, second pass: no phone; with curl as the app calls it, `server_sound=true` is answered as
    text/plain; the speaker's name and a barge-in were not seen.*
    *D: not done (no phone); the `server_sound` request is logged as answered as text.*

**E. Updates**

30. A nightly update end to end (or `restart_server` after a push to the followed branch): pull,
    `RequirementsChecker.sync` with the CPU index (torch stays `+cpu` and compares equal to the
    pin), restart through the unit, HEAD moved.
    *U: proven on both paths against a repository of the VM's own followed as `master`
    (`developer_release: true`): `restart_server` and the nightly update (the clock set to 03:00)
    each pulled a commit with a changed pin, ran pip with the CPU index, restarted through the unit
    on the new HEAD. The `release` line was not followed end to end — its tag cannot be signed
    there — but the real release's tag verifies on Linux, with `ssh-keygen`: `openssh-client` joined
    the apt packages for it. `+cpu`: see line 4.*
    *D: proven the same way on both paths (a pin up, a pin down; the nightly one by putting the
    VM in a time zone where it was 03:16): pulled, pip with the CPU index — torch untouched —,
    restarted through the unit on the new HEAD, 9 s each. The real release's tag (v2026.11.909)
    verifies with Debian's git 2.39 and OpenSSH 9.2, and a signers file with another key refuses
    it.*
31. A PR trial: `try_pull_request` restarts on `trial/pr-<N>`, the scoped `start.sh` watches the
    port; then a trial of a PR that cannot boot — the rollback to the previous branch, the note
    in `log/trial_rollback.txt` read from `log/raw-floxy.log`, the server back up.
    *U: `start.sh`'s part proven by hand, without `gh`: spawned inside the unit's control group
    with a fallback branch, it watched a trial branch come up (3 s), and for a branch that cannot
    boot went back after 16 s, left the note, and the relaunched server read it. `try_pull_request`
    itself: not tried.*
    *D: the same by hand, proven on systemd 252: the trial branch was seen up after 3 s; one that
    cannot boot was given up after 16 s, the checkout was back on the branch before and the server
    answering 21 s after the start, the relaunched server read the note and removed it.*
32. The nightly code-quality run finds `claude`, `gh` and `git` from the unit.
    *U: not tried (neither CLI installed); see line 10 for the PATH.*
    *D: not tried.*

**F. The Mac (M), before the release**

33. A fresh install with the readme's line: the same steps and texts as before this plan; the
    one changed command is the temp file of the log (`mktemp` with a full template) — install
    into a non-empty folder once to see the "Log:" line name a file that exists.
34. `KEEP_ROLE=imessage_satellite` installs a satellite as before.
35. `start.sh`, `stop.sh`, a `restart_server`, a nightly update, the watchdog and the Desktop
    shortcut on an existing install; the prompt and the setup toolkit render as before
    (`<!-- mac -->` passages kept, `<!-- linux -->` dropped).

## Order of work

Each step leaves `master` releasable for Macs.

1. **Platform.py, requirement markers, RequirementsChecker, CI on Ubuntu** (§1) — until the suite
   is green there. Mac behaviour unchanged.
2. **The switches** (§4) and the helpers (§6), tested under a patched `Platform.current` on the
   Mac — the voice split included: a test imports `VoiceInterface` with `sounddevice` refused and
   starts it as Linux.
3. **Scripts and the unit** (§2), tested through `KEEP_OS` and the staged process table.
4. **install.sh** (§3) and the texts (§5). The readme names Linux from here on, as new and in
   testing.
5. **A real install** on an Ubuntu 24.04 machine and a Debian 12 one, and a Mac: the checklist
   above. Steps 1–4 are done; of this one, Ubuntu 24.04 and Debian 12 on aarch64 are (the *U:* and
   *D:* notes).
6. Release; the readme drops "in testing" only after step 5.
