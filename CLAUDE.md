# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code
Claude: this codebase is yours - what it's for and how it's built are your call, and you'll keep building on it.

## Project Overview

Keep is a home automation voice assistant: a Python server (`floxy_ai_server.py`) providing an intelligent AI
assistant that the user can speak to, with multiple LLM backends (GPT, Claude, Gemini, Ollama, a local ds4-server, any
OpenAI-compatible local server such as oMLX or llama.cpp). This repository is the
server. Its clients are called "Outposts" — they act as "mics": local ones are microphones on the server machine,
remote ones (Android / iOS apps, not part of this repository) run on-device STT and send text to the server.

Voice is one way in. The others are chats — the Telegram bot, the terminal console (`tui.py`) and iMessage, where
the assistant is a contact of its own with its own Apple ID — and HTTP, which the apps speak.

## Git Workflow

- **Only commit when explicitly asked** — never commit automatically after completing work.
- **Always use the `/commit` command** when committing changes. Never commit manually.
- **A push to master succeeds.** The remote prints "Changes must be made through a pull request" on every
  push — it is a notice, not a rejection; check `git log origin/master` before concluding a push failed.
- **Never `git reset --hard`, `git checkout -- <file>` or `git stash` a tree with changes you did not make.**
  Other sessions work in this same checkout; their uncommitted edits are on disk and nowhere else.

## Releases

Development happens on `master`; installs track the `release` branch. `scripts/release.sh` fast-forwards
`origin/release` to `origin/master` and tags the commit with its version (`vYYYY.MM.N`, see
`src/common/Version.py`) — that is the whole release. The tag is signed (`git tag -s`, an SSH key in the
maintainer's 1Password) and an install moves to a release only when the tag on the tip of `release` verifies
against `res/release_signers` as the running code has it (`common/SignedRelease.py`); a new signing key is
added by a release signed with the old one. The readme's install command fetches `install.sh` from `release`;
with `KEEP_ROLE=imessage_satellite` in front of `bash` the same installer makes an iMessage satellite instead.
The same command installs a Linux server (Ubuntu, Debian — `docs/plans/LINUX_PLAN.md`; new, not yet proven on a
real machine): each Linux step stands beside the Mac's behind `linux` (`KEEP_OS` overrides `uname -s` for the
tests alone), the missing apt packages are the one thing installed with sudo (after an `apt-get update` where apt's lists do not know them), and `start.sh` runs the server as
the systemd user unit `keep.service` (`scripts/service.sh`) where the Mac has Terminal, the login applet and the
watchdog. A satellite is a Mac: the role is refused on Linux.

`restart_server` checks out the branch the `developer_release` config key names (false → `release`, true →
`master`, as on the development machine) and pulls it; `RequirementsChecker.sync` then runs pip when the
pulled pins no longer match the venv, so a release may bump a pin freely, and a failed install leaves the old
process running. The nightly update (`CodeQualityMaintainer._nightly_update`, `nightly_update` config key, on
by default) does the same at 03:00, before the nightly code-quality run, and restarts only when HEAD is no
longer the commit the process booted on. A server booted on any other branch that is not a PR trial runs a
branch of the install's own (`CodeQualityMaintainer.own_branch` — somebody develops on that checkout): the
nightly update skips it and `restart_server` relaunches on it without pulling, until `leave_own_branch` goes
back to the target branch. The memory curation is due from 04:00.

The machinery under both is `common/Updater.py` (switch, pull, requirements sync, relaunch through a start
script handed the process's own pid). It reads no config and imports nothing of the assistant, so the iMessage
satellite updates through the same code; `CodeQualityMaintainer` delegates to it and adds what is the server's
alone. Never hard-code `master` in a git command — `CodeQualityMaintainer._current_branch()` is the branch to
come back to after a Claude Code run, `target_branch()` the one to pull.

## Code Style

- Keep code concise and readable
- Keep it simple
- NO trivial comments - only add comments when they add real value
- Follow existing patterns in each codebase
- Python: Class files use uppercase naming (e.g., `VoiceSynthesizer.py` contains `class VoiceSynthesizer`)
- No `sys.platform` / `platform.system()` outside `common/Platform.py`: what differs between a Mac and a Linux server
  branches on `Platform.is_mac()` / `Platform.current()`, and its test patches `Platform.current` — so each OS's branch
  is tested on the other (`tests/unit/common/test_platform.py` fails on a direct test of the OS). A test of what only
  a Mac does (a login item, launchd, the local Messages host) takes the conftest's `on_a_mac` fixture, so it passes on
  the Ubuntu job too
- **What a Linux server leaves out** (`docs/plans/LINUX_PLAN.md`), each behind `Platform` and said to the user and the
  LLM rather than failing: the server's own microphones and speakers (`VoiceInterface.local_audio` — off the Mac nothing
  of `server_audio/` or of the local outposts is imported, a `server_sound` request is answered as text and
  `config_set` refuses `local_outposts.*`), an integration whose class sets `mac_only` (Photos, Reminders, HomeClaw:
  never constructed, a `MacOnlyIntegration` stand-in in the monitor, "needs macOS" in `list_integrations` and the Catalog
  tab, refused by `enable_integration`), a `local` Messages host (`Hosts.build` hands out a host that answers
  not-configured; iMessage needs a satellite) and the satellite role, the login item, the Desktop shortcut, the launchd
  watchdog and Full Disk Access (each entry returns early), `Commands.MAC_ONLY_COMMANDS`, and the Dashboard's GPU and
  temperature tiles. A prompt text says what differs between `<!-- mac -->` … `<!-- /mac -->` and `<!-- linux -->` …
  `<!-- /linux -->` lines (`formats/Rewriter.gate_platform`; `Toolkit.text()` and `FirstRun.brief()` apply it): each
  system reads its own passage, so a Mac's rendering is what the text said without the markers —
  `tests/unit/assistant/commands/formats/test_platform_passages.py` holds that

## Logging
Always add `info()` / `trace()` calls for important operations, decision points, and "nothing to do" 
outcomes (e.g., empty query results, skipped items). We are aiming for a system that's very easy to debug - good logging
helps with this!

## Development Commands

**Working directory**: the repository root — the server runs from it and cwd-relative names (`log/…`, `res/…`) assume it
**Python venv**: `.venv` in the repository root (`python3.12 -m venv .venv && pip install -r requirements.txt -r requirements-voice.txt`; on Linux the voice file goes in with `--extra-index-url https://download.pytorch.org/whl/cpu`, as `install.sh` and `RequirementsChecker.install` pass it, or pip brings torch's CUDA build)
**Config example**: `config.example.yaml`, plus `src/integrations/<folder>/config.example.yaml` for the section an integration owns — **always update the one that holds the key when adding/changing config keys**. `ConfigSchema` reads them as one text: a `# >> section: path` line in the root stands where the fragment's section belongs (a new fragment needs that pointer line — `tests/unit/common/test_config_schema_fragments.py` fails without it), and a fragment holds nothing but the section and the comments above it, since `config_help` shows it verbatim.

## Architecture

### Core Components

**Layout**: the root holds only the entry points (`floxy_ai_server.py`, `tui.py`, `keep.sh`, `start.sh`, `stop.sh`, `autostart.sh`, `install.sh` — called by path from outside the tree), `scripts/` (what only the tree or a person calls; each cds to the root first), config, data (`res/`, `cache/`, `log/`, `voice_id/`, `memory/`, `floxy.db`), `tests/` and `tools/` (standalone maintainer scripts the server never imports — `tools/imessage_live_check/` is the check a person runs at an unlocked Mac for what no test can show of the iMessage host); the four source packages live under `src/` and are imported without a `src.` prefix (`from common.x import …`): `pytest.ini` adds `src` via `pythonpath`, and `floxy_ai_server.py` and every external script insert `<root>/src` at the front of `sys.path`. `common/Paths.py` (`AISERVER_DIR`, `SRC_DIR`, `INTEGRATIONS_DIR`, `RES_DIR`, `CACHE_DIR`, `LOG_DIR`, `MEMORY_DIR`, `CONFIG_PATH`, …) is the one place that knows where the tree is — never do `Path(__file__).parents[N]` arithmetic in a package module.
- `floxy_ai_server.py` — thin launcher: `build(port, testsuite) -> Server` constructs the `Assistant` and starts every enabled interface; `main()` serves; `_launch()` is what runs under `__main__` — requirements check, migrations, then the role (`common/Role.py`): the server, or with `role: imessage_satellite` an iMessage satellite and no assistant at all (`_serve_satellite`); any other value starts nothing
- `src/common/` — logging, ConfigManager (which runs `ConfigMigrations` — the one-off moves of config.yaml sections, such as the mail and calendar accounts — before the sections are read), Paths, Platform (the one place that asks the OS what it is — `MAC` / `LINUX`, `machine()` and `mac_only(what)` for the wording; it imports nothing of the tree), DevelopmentConfig, FloxyDatabase, monitors, RequirementsChecker, SignedRelease, Updater, PrivateFile (every writer of config.yaml and of its backups goes through it: the file keeps its mode — the installer's 0600 — whatever the umask)
  - `imessage/` — what must run next to Messages.app, cut out of the iMessage integration so that two consumers — `integrations/imessage` (the owner's account: the feed, `send_imessage`, the 2FA codes) and `interfaces/imessage` (the assistant's own account: a chat) — and a satellite on another Mac or macOS user can use it. `docs/plans/IMESSAGE_HOST_PLAN.md` is the design, with the list of what has not yet been verified on a real Mac. The package imports nothing of `assistant`, `interfaces` or `integrations`.
    - **The host**: `MessagesHost` is the one interface to a Mac's Messages account (the reads, `send`, `send_chat` for a consumer that answers in the chat, `groups`, `typing(chat, on)`) and holds the dataclasses that cross it. It is made for a wire: every argument and result is JSON-ready (`Wire.to_wire` / `from_wire`), a failure is a value in the answer, an attachment crosses as an id with a call for its bytes, and the host keeps nothing per consumer — the cursor, the horizon and the owner's name are the caller's arguments. `LocalMessagesHost` is the in-process implementation over `MessagesDatabase` (chat.db), `Contacts`, `MessagesSender` (AppleScript) and `TypingIndicator`. `integrations/imessage` touches Messages only through `self.host`. A chat the assistant's own account is in is the chat interface's to read and never a feed row as well: while `interfaces.imessage.enabled` is on, the feed leaves out every polled row whose chat has one of that account's addresses (`HostHealth.addresses`) among its participants.
    - **Which host**: a consumer's config section (`imessage`, `imessage_chat`) has `host` (`local`, or `http://<address>:8898`) and `host_token`, and `Hosts.build(section)` is the one place that turns them into a host — the process's one `LocalMessagesHost`, or a `RemoteMessagesHost`. Both keys are typed by the user (`ConfigStore.OWNER_TYPED_PREFIXES`: `config_set` refuses to state them — an address read in a mail must never become where the token and the messages go). `RemoteMessagesHost` speaks JSON over HTTP (`Wire`: the routes, the limits, `WIRE_VERSION`); the calls are derived from the interface's abstract methods, so a call added to `MessagesHost` is on the wire on both ends; the token goes in the `Authorization` header and is in no URL, log line or error.
    - **The satellite** is the other end — a Keep install whose root config key `role` is `imessage_satellite` (`common/Role.py`): `floxy_ai_server._launch` then builds no `Assistant` and imports nothing of `assistant`, `integrations` or `interfaces` (`tests/unit/test_satellite_role.py` proves it), only `Satellite.build()` — the local host behind `SatelliteService` (a Quart app that answers nothing without the token) on `imessage_satellite.port`, bound to 127.0.0.1 unless `imessage_satellite.lan_access`. It has no nightly clock: every request names the server's branch and commit, and `ServerFollower` brings the install there through `common/Updater`. `role` and `imessage_satellite.*` are denied to `config_set` / `config_request` (`ConfigStore.DENIED_PREFIXES`).
    - **The typing dots** are made by setting the compose field of Messages through Accessibility, in a window of the chat's own that `TypingIndicator` opens and closes again; the host ends them by itself after `TIMEOUT_SECONDS`. Dots in the wrong chat are the one failure not allowed: anything short of certain ends in no dots, a cleared field and a closed window. The gates (Accessibility granted, the screen unlocked, Messages open …) are re-checked at every `on`; the one that is closed travels in `HostHealth.typing`. Every pyobjc call sits in `Accessibility.py`, refused by the unit conftest — the tests run over `tests/unit/common/imessage/ax_fakes.py`, and nothing may call the real Accessibility API against Messages outside `tools/imessage_live_check/`.
    - **The scripts follow the role**: `scripts/role.sh`, sourced by `start.sh`, `stop.sh`, `autostart.sh` and `keep.sh`, reads `role` and `imessage_satellite.port` from `config.yaml` the way the Python side does (`tests/unit/test_role_sh.py` holds the two readings together). A satellite usually shares a Mac with its server (a second macOS user), so every process lookup of those scripts goes through `own_listeners` / `own_processes` — never a bare `pgrep -f` or `lsof -i` in a script. The script tests stage a process table (`tests/unit/process_table.py`).
    - **The scripts follow the system too**: `role.sh` sets `os` (`uname -s`; `KEEP_OS` overrides it for the script tests alone) and holds the Linux forms of the lookups (`ss`, `/proc/<pid>/cwd`). `start.sh` keeps the head and the tail and sources the launch — `scripts/launch_mac.sh` (stop the old pids, Terminal) or `scripts/launch_linux.sh`: there the server is the systemd user unit `keep.service`, written and enabled by `scripts/service.sh` (`ensure_unit`; one Keep per user, a folder systemd's syntax would misread is refused), a start is `systemctl --user restart`, and a `start.sh` spawned by the server itself runs again in a `systemd-run --scope` so the unit's restart does not end it before the PR-trial watch. `stop.sh` stops the unit, then what is left; `autostart.sh` and `scripts/cli_start.sh` are the Mac's alone and refuse to run on Linux. "The Mac does not change" (`docs/plans/LINUX_PLAN.md`): a Linux behaviour is a branch beside the Mac's commands, and every script test runs both branches on both systems.
- `src/assistant/` — the assistant proper; it never imports `interfaces.*`
  - `Assistant.py` (facade: `stream(RequestContext) -> AssistantEvents`, handler pool), 
  - `commands/` (Commands — the core `cmd_*` commands and the dispatch; `CommandSet.py` — the base of the commands an integration folder brings, discovered and hosted by `Commands`; CommandProcessor; `Toolkits.py` — the command references the LLM loads on demand; `formats/` — the call syntaxes the LLM writes a command in (`keep`, `qwen`, `dsml`, `invoke`), each a `CommandFormat` the processor parses through and `PromptManager` renders the prompt's declarations and examples in — one per process, `CommandFormats.active()`, chosen by the smart tier's LLM class or the tier's `command_format:` config key; `RescueFormat.py` reads a call written as an `<invoke>` block under another format. The texts follow one convention (`docs/plans/FORMAT_NEUTRAL_TEXTS_PLAN.md`): a mention is the bare backticked name, `name()` is only the declaration of a parameterless command, the prose states facts about values and never how to type them, and prose only Keep's syntax needs stands between `<!-- keep -->` and `<!-- /keep -->` lines — `tests/unit/assistant/commands/formats/test_texts_are_format_neutral.py` fails otherwise)
  - `prompt/` (PromptManager — assembles the prompt from `res/prompt.md` and the live sections; `PromptSectionGenerator.py` — the base of a section and the discovery of the ones integration folders bring; `sections/` — the core's own generators; MessageHistory)
  - `agents/CodeQualityMaintainer.py` runs the Claude Code CLI on the repo: the nightly run (test generation — a tests-only PR merged on the spot, a bug a new test exposes pinned as a strict xfail and listed, never fixed there —, bug analysis of the day's commits, log triage, a PR fixing the Critical and Major items, test health — the suite measured idle for the slow tests and for the flaky ones beside six busy loops per core, the load under which it is as slow as on a CI runner (`scripts/test_under_load.sh`: an idle machine never shows a test that depends on real time), once more under load with `config.example.yaml`, the config CI and a fresh clone run on (the install's `config.yaml` never shows a test that depends on a value of it), and the day's failed runs of the CI workflows read for the tests that fail on the runners alone —, auto-merge, and last the open pins — every strict xfail of the suite, every night until the maintainer fixes the bug or marks the test `@pytest.mark.wontfix`, which only a human adds), the nightly update before it and the commands of the `developer` toolkit — `self_analysis`, `improve_integration` (a PR confined to one integration folder), `report_issue`, `try_pull_request` / `pr_feedback` (`agents/PullRequestTrial.py` — the server runs one of its own PRs as `trial/pr-<N>` until the PR is merged and released — the nightly update keeps the trial and merges each new release into it — or the next `restart_server`) — and `self_improvement` (a PR anywhere), offered only while the root config key `is_developer` is true. Every run is on a Fable or an Opus model (`build_model()`): `config_set` refuses any other for `ai.code_quality.model`, a config that names one runs on Opus, and Opus is what a text names as the way out when Fable is out of reach — never "a cheaper model".
  - long-term memory, all under `memory/` (gitignored): `MemoryFile.py` — the bank `memory/bank.md` (whole in every prompt; `remember` appends, the curator rewrites it; the prompt shows the curated part at the top and the notes appended since in a block lower down — `MemoryFile.split`, so a `remember` never changes the cached top of the prompt); `WatchList.py` — `memory/watchlist/<slug>.md`, one file per tracked item (the `watch` toolkit, `res/toolkits/watch.md`; the prompt carries a one-line-per-item index and documents `load_watch` alone, which brings the toolkit along); `Procedures.py` — `memory/procedures/<slug>.md`, the how-tos (`load_procedure`, `save_procedure` / `delete_procedure`; the prompt carries only a code-built index). `memory/` holds only what is live: replaced and deleted versions are copied to `log/memory/`, a debugging trail the LLM never reads.
  - `agents/LongTermMemoryCurator.py` — the nightly run: rewrites the bank, then tidies the watch list and may add (never replace or delete) procedures; a curated bank more than a tenth over `MEMORY_WORD_LIMIT` is then condensed (`MemoryCondenser`): a pass asked down to `CONDENSE_AIM` (an eighth) under the limit, since the model delivers a part of its ask, and again while the bank is over the trigger — three passes at most, and a pass that removes under a tenth of its ask ends them with an error. The nightly run goes through the batch API at half price (`ai.memory_curator.batch`): the curation as one batch, the condensing one batch per section, one after another, and the first section whose batch gives no answer within `MemoryCondenser.BATCH_WAIT_SECONDS` sends the rest through ordinary calls. `//memory_curate` runs a curation over given days, `//memory_condense` the condensing alone.
  - self-setup: `list_integrations` / `config_help` / `config_request` / `restart_integration` let the assistant configure an integration itself; a folder whose accounts sign in brings the command for it (`microsoft_login`, `google_login`), listed in the `setup` toolkit. The setup also registers `Keep Autostart.app` (`common/Autostart.py`, an applet running `autostart.sh`) as a login item, so the server starts at login through `start.sh` and Terminal, where the Mac's grants apply. The root config key `watchdog` (off by default; `common/Watchdog.py`) adds a launchd agent that opens the same applet every 5 minutes — launchd runs only `open`, never a shell in the checkout, since a checkout under ~/Documents would hang a launchd shell on a consent dialog (#127). All of this is the Mac's: on Linux `Autostart.enable`, `Watchdog.sync` and `DesktopShortcut.ensure` return early, and the systemd unit (`scripts/service.sh`: enabled at every start, `Restart=on-failure`) does the starting at boot and the reviving.
  - `integration/` — the framework for outside services: `Integration`, `IntegrationMonitor` (a `check_health` that raises `HealthWarning` is shown as WARNING and never counted towards ERROR), `Feed` / `FeedRunner` (a feed that raises `FeedSourceAway` — its source out of reach, not broken — keeps its cursor and is asked again instead of failing after `FAILURE_LIMIT`), `bootstrap.py` (discovery), `SatelliteHints.py` (what a problem with the Mac that holds a Messages account means to the owner) and the contracts more than one integration implements (`HomeAutomation` with `CompositeHome` — `HomeAutomation.active()` —, `CameraProvider` — `CameraProvider.active()` —, `VehicleProvider` with `Fleet` — `VehicleProvider.fleet()` —, `TfaCodeHandler`, under `music/` the speaker contract `MusicSystem` with `Speakers` — the zones of every system as one house, the `play`/`pause`/`play_music` … commands and the MUSIC and zones sections resolve a room through `MusicSystem.speakers()`; a system's own list ids carry its `ID_PREFIXES` (Sonos `F-`, `P-`), Spotify's (`S-n`, the search ids) are any system's — and the household's one Spotify account every system shares (`SpotifyAccount`: search and its short ids, the playlist library, the sign-in — `SpotifyAuth`, whose scopes grow by `want_playback()` when a system starts Spotify over Connect), and under `mail/` and `calendar/` the account contracts with their composites: `MailAccount` / `ImapMailAccount` / `MailSource` / `MailAccess` and `CalendarAccount` / `Event` / `CalendarSource` / `CalendarAccess`). An integration folder implements `MailSource` / `CalendarSource` and holds one account per entry of its `<section>.accounts` list (`Accounts.py`); everything else goes through `MailAccess.current()` / `CalendarAccess.current()`, which gather the accounts of every enabled and configured integration through the monitor — never through a folder by name.
- `src/integrations/` — one folder per outside service (`tesla/`, `icloud/`, `photos/`, `imessage/`, `whatsapp/`, `home_assistant/`, `imap/`, `caldav/`, `google/`, `microsoft/` — the mail and calendar accounts, …) holding everything that belongs to it; the core finds each piece by discovery, so adding an integration is adding a folder. **The file suffix is the role, and each side loads only its own** — everything but `*Tab.py` is the assistant's and never imports `interfaces.*`:
  - `<Name>Integration.py` — the `Integration` class (`assistant/integration/bootstrap.py` walks the folders); helper modules beside it.
  - `<Name>Commands.py` — its commands, a `CommandSet`: `cmd_<name>` / `describe_<name>` methods and the flags of its own commands (`MULTI_PARAM_COMMANDS`, `SENSITIVE_COMMANDS`, `PURE_COMMANDS`, `APPROVAL_NOTES`). It reaches the shared machinery through the host (`self.background_command`, `self.profile`, `self.context`) and a helper that stays in core through a public member of the host (`self.commands.integration_monitor()`, `first_argument`, `media_tag`, …), never a private one. `Commands` hosts every discovered set whether its integration is enabled or not and refuses a command name with two owners; `Commands.command_names()` is the full list without an instance. A command stays in `Commands.py` when more than one integration is behind it (home, mail, calendar, cameras, cars, music) or none.
  - `<Name>SectionGenerator.py` — its prompt data, a `PromptSectionGenerator` declaring `key` and `heading` (its own heading line, hashes included, plus an optional `intro`): the section stands under that heading in the prompt's one `{INTEGRATION_DATA}` block (the template never names an integration), or, with `sections:` in `toolkit.md` instead of a heading, is carried by the toolkit (`load_toolkit` then sends the data). `PromptManager` runs the discovered generators beside the core's (`prompt/sections/`: memory, procedures, watch list, vault, home, cars, calendar, music) and refuses a key with two owners. **A folder's generator runs only while the folder's integration is on**: `PromptManager` follows the `IntegrationMonitor`'s verdict and builds, starts, stops and drops the gated generators as it changes — a core generator joins the gate with `needs = <contract>` (home, cars, calendar, music). A generator never checks whether its integration is enabled or waits for it to be built.
  - `<Name>Tab.py` — its console sub-tab, an `IntegrationTab` naming its `integration` class; on the Integrations row while that integration is enabled.
  - `toolkit.md` — the LLM's command reference when the folder alone owns the toolkit: front matter (`name`, `summary`, `commands`, `sections` / `hidden` when set) over the text; `assistant/commands/Toolkits.py` globs for it. A toolkit is offered only while its integration is on (`Toolkit.offered_by`, the monitor's verdict; a core toolkit fronting a contract declares `needs=` — home, cars, music, email, calendar): off, it is out of the `{TOOLKITS}` index, `load_toolkit` and the preloads answer that the integration is off, and `Commands.run` refuses its commands with the same answer. A disabled integration is visible to the LLM nowhere but `list_integrations`.
  - `config.example.yaml` — its config section with the how-to comments, read into the one schema by `common/ConfigSchema.py` (see Config example above).
  - `requirements.txt` — pins of packages only this folder imports, included from the root file with `-r`.

  **Local integrations**: an install's own integration folders can live outside the checkout — `extra_integration_dirs` in `config.yaml` lists directories laid out like `src/integrations/` (`~/KeepLocal` holding `schoolsoft/`, `podcasts/` …). `common/IntegrationDirs.py` resolves them; `src/integrations/__init__.py` appends them to the package's `__path__`, so a local folder is `integrations.<folder>` to every module discovery, and the file globs (toolkits, config fragments, `requirements.txt`) go through `IntegrationDirs.files()`. A local folder named like an earlier one is shadowed, logged and left out. The unit conftest pins the key to `[]`, so no test sees them.

  A broken folder costs only itself: a file discovery cannot read or import is logged and left out (for the folders of this repository the unit suite fails instead). Core reaches a folder through a contract or discovery, not by name — `tests/unit/test_integration_folders.py` holds the allowlists of the by-name imports that are left (core → folder, folder → folder, the launcher); shrink them, never grow them.

  **Adding an integration**: a folder with `__init__.py`; `<Name>Integration.py` whose class docstring has three paragraphs — what it is in a few words (the atlas description and the "Description" column of the console's Catalog tab), what it does for the user in one to three sentences (shown with the integration selected there), then the notes for whoever reads the code, which no UI shows — and whose constant `name` is what `enabled_integrations` lists; whichever of the pieces above it needs; a `GROUPS` blurb in `docs/atlas/descriptions.py`; tests in `tests/unit/integrations/<folder>/`. A folder that keeps a signed-in session in the tree lists its files in `wipe_targets()` and ends the session on the account in `log_out()` (`//wipe` gathers both by discovery); one with a login command of its own names it in `login_hint`. What is part of the prompt has a pinned order — give a new toolkit its place in `Toolkits.INDEX_ORDER` and a new section its place in `PromptManager.SECTION_ORDER`.
- `src/interfaces/` — one package per way of reaching the assistant, each an `Interface` (`enabled(config)`, `start(assistant, app)`, `start_async()`, `stop()`); a disabled interface's module is never imported
  - `http/` — HttpInterface (Quart app, `/stream`, `/health`, hypercorn, signals), `routes.py`, `WireFormat.py`
  - `voice/` — VoiceInterface (whole VoiceToVoice stack: microphones, TTS, server audio, speaker id, iOS overlay; on a server that is not a Mac only the phones' side of it — no local microphones, speakers or Outpost TCP server), `VoiceTurnGate.py`
  - `chat/` — the chat contract every chat UI shares: `ChatTransport` ABC + `Draft` protocol (what a platform implements). A question only the owner answers (an approval, a secret) is asked in the transports' `owner_conversations()`; a turn from any other conversation has it asked there instead, and its own transport is told (`ChatTransport.asked_elsewhere`). `IncomingMessage.context` is what a transport that reads a whole chat puts in front of a turn, outside the sender's own words
  - `telegram/` — TelegramInterface + `TelegramTransport` (the bot, a thin `ChatTransport`: update parsing, `TelegramDraft` send/edit/flood handling, status line, approval keyboards)
  - `imessage/` — IMessageInterface + `IMessageTransport` (`interfaces.imessage.enabled`, off by default; listed by the monitor as "iMessage Chat"): the assistant as an iMessage contact of its own, a `ChatTransport` over a `MessagesHost` (`imessage_chat.host`) whose Messages account is the assistant's Apple ID — never the account the iMessage integration reads (both on one account: the chat refuses to start). It polls the host with a cursor of its own (`ChatState`, `cache/imessage_chat.json`) and serves the handles of `imessage_chat.owners` / `guests` / `bots` (`Allowlist`; every entry is typed by the user — `config_set` refuses them). A 1:1 message is always a turn. In a group the wake rule (`assistant/GroupWake.py`, shared with the Telegram feed) says which messages are addressed to it, and with `imessage_chat.listen_in_groups` (on by default) every other worded message of a listed person who is not a bot is a turn too, marked `IncomingMessage.overheard`: a bare NO — never sent — is the normal answer, and an overheard message never supersedes a reply in flight. What does not become a turn is the backlog put in front of the next one as untrusted context. Nothing somebody outside the household chose reaches the model as written (`Untrusted.py`), and a group is a `post_message` destination only once an owner has written in it. Replies go to the chat's guid as plain bubbles (`PlainText`), capped per recipient per day; no prompt is ever shown there — `owner_conversations()` is empty, so approvals and secrets go to the console and Telegram. While a reply is worked on the chat sees the account typing (`imessage_chat.typing_dots`): best effort, told to the host on a thread of its own, and a reply never depends on them. The setup the assistant walks the owner through is the comment over `imessage_chat` in `config.example.yaml` (and over `imessage_satellite` for the satellite's install).
  - `console/` — ConsoleInterface + `ConsoleTransport` (the TUI's chat, a `ChatTransport` rendering into the Home tab's `ChatPane`) + entire TUI. The Integrations tab is `tabs/IntegrationsGroup.py`: its sub-tab row is not a list in code but follows the `IntegrationMonitor` — `tabs/CatalogTab.py` is always first and lists every integration, and after it the discovered `*Tab.py` of each enabled integration. `tabs/AuthTab.py` is the sign-in board a login tab extends.

**Request Flow**: `interfaces/*` build a `RequestContext` → `Assistant.stream()` → `RequestProcessor` → `DialogManager` → LLM; commands run through `assistant/commands/CommandProcessor` and each round's result is persisted as a `command` history row (`MessageHistory.add_command_result`, rendered verbatim on the user side so the next turn still sees it); the events stream back to the interface (HTTP renders them with `WireFormat`; the chat interfaces through `interfaces/chat/ChatSession` → the transport's `Draft`).

**Destinations**: a reply goes back to whoever asked; `post_message(destination, text)` says something somewhere else. `src/assistant/Destinations.py` is the registry the interfaces fill at start — the chat hub registers `owner`, the Telegram transport one destination per allowed group, the iMessage transport one per person on its lists and per group chat an owner has written in, the voice interface the server's speakers under their room's name — and the `{DESTINATIONS}` line of the prompt follows it. The autonomous check emits commands only and speaks through `post_message`; stray text after its commands is dropped.

## Testing Strategy

Unit tests are pytest under `tests/unit`.

Before discovering test coverage gaps, verify findings by checking ALL test directories in tests/ (including 
integration/, slow/ etc). Do not report a module as untested without searching for its name across the entire test
tree.

**Test data is fictional — the unit tests are public.** Never copy a value from `config.yaml`, `memory/`,
`floxy.db`, the vault, the logs or a real conversation into a test, a fixture or a comment; invent one that
exercises the same code path. Use the standing cast and nothing from the real household:
- people: the owner **Marcus** (the unit conftest pins `owner_name` to it), family **Charlotte**, **Frances**,
  **Ingeborg**; outsiders Alice, Bob, Carol, Dave
- phone `+46701740605` (the 070-174 06 05…99 range is reserved for fiction), mail `m@mlab.test` or
  `*@example.com`, plate `KEP 47L`, personal id `19850203-1234`, hosts `*.local` / `*.example`, IPs from
  `192.0.2.0/24` or `100.64.0.0/10`, the house in `tests/unit/data/home/` (synthetic — regenerate its
  `*rendered.md` with the real renderer, never by hand)
- a test must pass with any `config.yaml`, the example one included (`FLOXY_CONFIG=config.example.yaml`): the
  conftest's `CAST_CONFIG` pins `owner_name`, the wake word keys and `ai_message_footer` to the cast's values,
  `location.home.timezone` to `Europe/Stockholm` (unpinned, a config with coordinates has the zone looked up from
  them — seconds on a CI runner, inside whichever test asks first), `extra_integration_dirs` to `[]`, `imessage.host` and `imessage_chat.host` to `local` (no test reaches an
  install's real satellite) and `interfaces.imessage.enabled` to false; patch any other config value a test
  depends on

**Tests not to write.** A test must assert something the author had to *derive*, never something copied from the
line it checks. Do NOT write a test that:
- asserts a constant, default or attribute equals its own definition (`assert DEFAULT_X == "xxx"`, `SUPPORTS_Y is True`)
- only checks a constructor stored its argument (`assert obj.llm is mock_llm`)
- mocks the collaborator and then asserts the mock handed back what it was told to (mock config in → same literals out)
- compares a user-facing string byte for byte (log lines, notices, help text) — assert the fact it carries, not the wording
- depends on the real clock (a sleep or a short timeout before a positive assertion, an assertion on elapsed time) — run the thread's body or fire the timer yourself, fake the clock of the module under test, and bound a wait that has to be real with `FAILURE_BOUND` / `wait_until` from `tests/unit/waiting.py`, never a literal

The exception is a value that is a contract with something outside the repo (a `config.yaml` key, a wire-format field
an Outpost parses, a path another system reads): pin it, and say in the test name what it is a contract with.

**Always run unit tests** — they complete swiftly:
```bash
source .venv/bin/activate && python -m pytest tests/unit --override-ini="log_cli=false" -q --tb=short 2>&1 | grep -E "^(FAILED|ERROR) |^=+ .*(passed|failed|error)"
```

**When to run them — the "code first, tests in the background" flow.** The suite is cheap; stopping mid-implementation to chase every failure is not. So:
1. Implement the whole change without touching or running tests.
2. When the code is done, run the unit suite **once** and report to the user: what was built plus the count and shape of the failures.
3. Spawn a **forked** background agent (`subagent_type: "fork"` — it inherits the conversation, so it knows what changed and why) to fix the failures. Keep taking new work meanwhile; relay the agent's result when it reports.
4. The agent edits files under `tests/` **only**. If a failure looks like a real bug in the new code, it reports that back instead of bending the test to match the bug.
5. While the agent runs, avoid editing the files its failures point at, and say so if asked for changes there — two editors on one file clobber each other.

Skip the background agent when the failures are few and trivial (a renamed symbol, a changed log line): fixing those inline is faster than spawning. The agent dies with the session, so if the user is about to leave, fix inline instead.

**Why these flags?** `pytest.ini` runs the suite on 8 xdist workers (`--dist worksteal`) with `log_cli=true` and `-s`; `tests/unit/conftest.py` holds the speed tricks (collection ordered by the previous run's durations, GC off during collection, SQLite without fsync, `sounddevice` stubbed). **It also keeps the suite off the machine it runs on**, and whatever adds a way of reaching the real Mac adds its refusal there (`tests/unit/test_no_server_relaunch.py` holds each one): the `Popen` guard refuses what `common/Updater` does for real (a script handed the test process's pid, a git command that moves the checkout, a pip install — so a test of anything that updates fakes git, pip and the start script), an AppleScript naming Messages or a login item, every `launchctl` call, `osacompile`, the Linux desktop tools (`xdg-open`, `wl-paste`, `xclip`) and systemd (`systemctl`, `systemd-run`, `loginctl` — the script tests run the scripts over bash functions of those names, and write the unit under a tmp `XDG_CONFIG_HOME`); `sqlite3.connect` refuses the Mac's own Messages and AddressBook databases; the pyobjc loader of `common/imessage/Accessibility.py` is replaced by a refusal. The wall time is ~7 s, about half of it the workers importing the tree — keep the run phase lean: no real sleeps or timeouts above ~50 ms in a test, and nothing that touches the real machine (Photos library, audio).

The suites outside `tests/unit` (`tests/prompt`, `tests/memory`, …) and `tests/unit/private` are **not part of this
repository**: they are built from a real household's data and are linked in (gitignored symlinks) on the maintainer's
machine only, together with `testsuite_prompt.md`, the prompt those suites run against. When they are present: they are
LLM/prompt-related and slow — only run them when changes affect LLM communication or prompts:
```bash
source .venv/bin/activate && python -m pytest tests/ --override-ini="log_cli=false" -q --tb=short 2>&1 | grep -E "^(FAILED|ERROR) |^=+ .*(passed|failed|error)"
```

**NOTE**: The prompt/integration tests spin up real server instances on fixed ports (8801+). Only one test suite run at a time per machine — concurrent runs will collide on ports and cause spurious "killed" failures.

## Key Files

- `floxy_ai_server.py` - Launcher (`build()` + `main()`; `_launch()` picks the role — the server or an iMessage satellite)
- `src/assistant/Assistant.py` - The facade every interface calls
- `src/assistant/DialogManager.py` - Core dialog logic
- `src/assistant/RequestProcessor.py` - One request: prompt → LLM → commands → history
- `src/assistant/integration/Integration.py` - The integration contract; `IntegrationMonitor.py` registers and health-checks them; the implementations are the folders of `src/integrations/`
- `src/assistant/integration/mail/MailAccess.py`, `calendar/CalendarAccess.py` - The one mailbox and the one calendar, gathered per call from the accounts of every enabled integration
- `src/interfaces/Interface.py` - The interface contract; implementations in `src/interfaces/{http,voice,telegram,console,imessage}`
- `src/interfaces/chat/ChatTransport.py` - What a chat platform implements; `ChatSession.py` is the shared reply/intake logic; `ChatHub.py` the Notifier
- `src/assistant/GroupWake.py` - The group wake rule the Telegram feed and the iMessage chat transport both decide by
- `src/common/imessage/MessagesHost.py` - The one interface to a Mac's Messages account; `LocalMessagesHost.py` in-process, `RemoteMessagesHost.py` over HTTP to a satellite; `Hosts.py` builds the one a config section names. Design and open points: `docs/plans/IMESSAGE_HOST_PLAN.md`
- `src/interfaces/console/ConsoleTransport.py` - The TUI as a chat transport; `widgets/ChatPane.py` is the Codex-style pane it draws into; `HistoryFeed.py` puts the dialog history in the same pane
- `log/floxy.out` - Log file from latest run on the server GREAT for DEBUGGING!
- `res/prompt.md` - The prompt template sent to the AI; the command references are the toolkits — `res/toolkits/<name>.md` for the ones more than one integration is behind (or none), `src/integrations/<folder>/toolkit.md` for the one an integration owns — only the core commands are in the template, and the integrations' data stands in its one `{INTEGRATION_DATA}` block, never under a heading of the template's own
- `testsuite_prompt.md` - (private, present only when linked) the prompt the prompt test suite runs against. It is built from `res/prompt.md` by the private tree's `testsuite_prompt/build.py`, so it is never edited by hand: after a change to `prompt.md`, run the builder (a private unit test fails until it is run)

## Architecture Atlas

`docs/atlas/` builds the Keep Atlas, the zoomable architecture map the readme links to (`python3 docs/atlas/build.py`,
published to https://keep-ai.dev/atlas/ on every push to master). A module's description is the first sentence of its docstring (its main class's when it has one), so a
new module needs a docstring that opens with one plain sentence. `docs/atlas/descriptions.py` holds the package blurbs,
overrides for missing or overlong docstrings, and the "Follow a request" tour; **add a package blurb to `GROUPS` when you
create a package, and update `TOUR` when a module it names moves**. `tests/unit/test_architecture_atlas.py` fails on
either omission.

## Public repository

The repository is public: every commit, branch, tag, issue and PR body is public the moment it reaches GitHub — not
only the tree. The Pages site is keep-ai.dev; it holds the Keep Atlas alone and its root redirects to the repository.

## Dependencies

- **Requirements files**: `requirements.txt`, `requirements-voice.txt` (the voice stack), and `src/integrations/<name>/requirements.txt` for what only one integration needs — included from `requirements.txt` by `-r`, so installing the root file installs them all (`tests/unit/common/test_requirements_checker.py` fails for a folder file the root does not include)
- A pin only the Mac can install or use carries `; sys_platform == "darwin"` (pyobjc, the server's own audio: `sounddevice`, `mlx-whisper`, `silero-vad`, `lingua-language-detector`); `RequirementsChecker` skips a pin whose marker is false, and on Linux passes pip PyTorch's CPU wheel index. `.github/workflows/linux.yml` installs both files and runs the unit suite on Ubuntu (x86_64 and arm64) — the base file must stay free of torch. `.github/workflows/mac.yml` is its counterpart on clean macOS 15 and 26 runners (Apple Silicon), with the install `install.sh` runs on a Mac — the voice file and no Homebrew package — so a test that passes only on the development machine fails there
- **IMPORTANT**: Always activate venv before running Python: `source .venv/bin/activate`
- A script importing `assistant`, `common` or `interfaces` needs a config: `ConfigManager` resolves `$FLOXY_CONFIG` if set, else `config.yaml` in the root
- System: Python 3.12; brew `mpv`, `portaudio`; Porcupine wake word models; Spacy model `en_core_web_sm`
- System, Linux: Python 3.12 (the distribution's with `python3.12-venv`, else uv's); apt `git curl ca-certificates ffmpeg iproute2 procps libsndfile1 libmagic1 openssh-client` — `APT_PACKAGES` in `install.sh`; no PortAudio or `mpv`, since a Linux server opens no audio device
