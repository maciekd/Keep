# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Floxy AI is a home automation voice assistant: a Python server (`floxy_ai_server.py`) providing an intelligent AI
assistant that the user can speak to, with multiple LLM backends (GPT, Claude, Gemini, Ollama, a local ds4-server, any
OpenAI-compatible local server such as oMLX or llama.cpp). This repository is the
server. Its clients are called "Outposts" — they act as "mics": local ones are microphones on the server machine,
remote ones (Android / iOS apps, not part of this repository) run on-device STT and send text to the server.

The system uses a predefined wake word, uses on-device and cloud speech-to-text, then sends the user query as text to 
the server. The server uses an LLM to process the query and can either play back the audio through local speakers or 
send the stream back to the client for playback.

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
`src/common/Version.py`) — that is the whole release. The tag is signed (`git tag -s`, an SSH key
in the maintainer's 1Password, set with `git config --local` in the development checkout) and an
install moves to a release only when the tag on the tip of `release` verifies against
`res/release_signers` as the running code has it (`common/SignedRelease.py`; every pull of
`release` goes through it) — a push to `release` without the key reaches no install, and a new
signing key is added by a release signed with the old one. The readme's install command fetches
`install.sh` from `release` and the installer clones it; with `KEEP_ROLE=imessage_satellite` in front of
`bash` the same installer makes an iMessage satellite instead (see `common/imessage/` below: no API-key
step, a small `config.yaml` with the role, a generated token and `watchdog: true`, and a summary with the
lines the Keep server needs to pair), and an existing install keeps the role its `config.yaml` names.
`restart_server` checks out the branch the
`developer_release` config key names (false → `release`, true → `master`; the assistant may flip it with
`config_set`) and pulls it, so an install picks a release up on its next restart and a machine switches
line by flipping the key and restarting. The development machine's `config.yaml` has it true. Between the
pull and the restart `RequirementsChecker.sync` runs pip when the pulled pins no longer match the venv,
so a release may bump a pin freely; a failed install leaves the old process running and says why. The
nightly update (`CodeQualityMaintainer._nightly_update`, `nightly_update` config key, on by default) does
the same checkout and pull at 03:00 — before the nightly code-quality run when that is on; a process that
boots inside that hour takes the night over — and restarts only when HEAD is no longer the commit the
process booted on. The memory curation is due from 04:00: the first periodic tick after that hour with
no dream journal written since 04:00 runs it, so a restart never loses the night.

The machinery under both is `common/Updater.py`: the branch of the checkout, the switch and pull (`release`
through `SignedRelease`), the requirements sync, the verdict (`update` restarts only when HEAD left the boot
commit, `refresh` whatever the pull brought; an `Update` carries the failure, what changed and what was
installed) and the relaunch through a start script handed the process's own pid. It reads no config and
imports nothing of the assistant — the target branch and the start script are arguments — so a process
without an assistant updates through the same code: the iMessage satellite does, following the branch and
commit of the server that calls it. `CodeQualityMaintainer` keeps the names its callers use as delegates and
adds what is the server's alone: the two config keys, the nightly lock, the system notes, the resume marker,
the PR trial's fallback branch. Never hard-code `master` in a git command —
`CodeQualityMaintainer._current_branch()` is the branch to come back to after a Claude Code run,
`target_branch()` the one to pull.

## Code Style

- Keep code concise and readable
- Keep it simple
- NO trivial comments - only add comments when they add real value
- Follow existing patterns in each codebase
- Python: Class files use uppercase naming (e.g., `VoiceSynthesizer.py` contains `class VoiceSynthesizer`)

## Logging
Always add `info()` / `trace()` calls for important operations, decision points, and "nothing to do" 
outcomes (e.g., empty query results, skipped items). We are aiming for a system that's very easy to debug - good logging
helps with this!

## Development Commands

**Working directory**: the repository root — the server runs from it and cwd-relative names (`log/…`, `res/…`) assume it
**Python venv**: `.venv` in the repository root (`python3.12 -m venv .venv && pip install -r requirements.txt -r requirements-voice.txt`)
**Main server program**: `floxy_ai_server.py`
**Config location**: `config.yaml`
**Config example**: `config.example.yaml`, plus `src/integrations/<folder>/config.example.yaml` for the section an integration owns — **always update the one that holds the key when adding/changing config keys**. `ConfigSchema` reads them as one text: a `# >> section: path` line in the root stands where the fragment's section belongs (a new fragment needs that pointer line — `tests/unit/common/test_config_schema_fragments.py` fails without it), and a fragment holds nothing but the section and the comments above it, since `config_help` shows it verbatim.

**Running Python scripts**:
```bash
# Run from the repository root since imports are relative to it:
source .venv/bin/activate && python <script.py>
```

## Architecture

### Core Components

**Layout**: the root holds only the entry points (`floxy_ai_server.py`, `tui.py`, `keep.sh`, `start.sh`, `stop.sh`, `autostart.sh`, `install.sh` — scripts macOS, a running install or the readme's one-liner call by path), `scripts/` (the ones only the tree or a person calls: `cli_start.sh`, `release.sh`, `migrate.sh`, `local_*_start.sh`; each cds to the root first — and `role.sh`, which the root scripts source), config, data (`res/`, `cache/`, `log/`, `voice_id/`, `memory/`, `floxy.db`), `tests/` and `tools/` (standalone maintainer scripts the server never imports, one folder each — `tools/imessage_live_check/` is the check a person runs at an unlocked Mac, with a handle of their own, for what no test can show of the iMessage host: the typing dots' gates, the chat.db columns, dots on a real phone); the four source packages live under `src/` and are imported without a `src.` prefix (`from common.x import …`). `pytest.ini` adds `src` via `pythonpath`; `floxy_ai_server.py` and every external script insert `<root>/src` at the front of `sys.path`. `common/Paths.py` (`AISERVER_DIR` — the repository root, `REPO_DIR` — the same, `SRC_DIR`, `INTEGRATIONS_DIR`, `RES_DIR`, `CACHE_DIR`, `LOG_DIR`, `VOICE_ID_DIR`, `MEMORY_DIR`, `CONFIG_PATH`) is the one place that knows where the tree is — never do `Path(__file__).parents[N]` arithmetic in a package module.
- `floxy_ai_server.py` — thin launcher: `build(port, testsuite) -> Server` constructs the `Assistant` and starts every enabled interface; `main()` serves; `_launch()` is what runs under `__main__` — requirements check, migrations, then the role (`common/Role.py`): the server, or with `role: imessage_satellite` an iMessage satellite and no assistant at all (`_serve_satellite`); any other value starts nothing
- `src/common/` — logging, ConfigManager (which runs `ConfigMigrations` — the one-off moves of config.yaml sections, such as the mail and calendar accounts — before the sections are read), Paths, DevelopmentConfig, FloxyDatabase, monitors, RequirementsChecker, SignedRelease, Updater
  - `imessage/` — what must run next to Messages.app, cut out of the iMessage integration so that two consumers — `integrations/imessage` (the owner's account: the feed, `send_imessage`, the 2FA codes) and `interfaces/imessage` (the assistant's own account: a chat) — and a satellite on another Mac or macOS user can use it. `docs/plans/IMESSAGE_HOST_PLAN.md` is the design, with the list of what has not yet been verified on a real Mac. The package imports nothing of `assistant`, `interfaces` or `integrations`.
    - **The host**: `MessagesHost` is the one interface to a Mac's Messages account (`health`, `cursor_at`, `remaining`, `poll`, `messages`, `scheduled`, `attachment`, `resolve`, `send`, and for a consumer that answers in the chat `send_chat` — a chat's guid or a handle, nothing resolved, on a send budget of its own —, `groups` and `typing(chat, on)`) and holds the dataclasses that cross it. It is made for a wire: every argument and result is JSON-ready (`Wire.to_wire` / `from_wire`), a failure is a value in the answer, an attachment crosses as an id with a call for its bytes, and the host keeps nothing per consumer — the cursor, the horizon, the tracked rows and the owner's name are the caller's arguments. `LocalMessagesHost` is the in-process implementation over `MessagesDatabase` (the chat.db reads), `Contacts` (the AddressBook cache, handle ranking, recipient and group resolution), `MessagesSender` (the AppleScript sends, the rate limits, the missing Automation permission) and `TypingIndicator`. `integrations/imessage` is the assistant's side of the owner's account (history rows, injection screening, media descriptions, 2FA forwarding, send-later tracking, system notes) and touches Messages only through `self.host`.
    - **Which host**: a consumer's config section (`imessage`, `imessage_chat`) has `host` (`local`, or `http://<address>:8898`) and `host_token`, and `Hosts.build(section)` is the one place that turns them into a host — the process's one `LocalMessagesHost`, or a `RemoteMessagesHost`. Both keys are typed by the user (`ConfigStore.OWNER_TYPED_PREFIXES`: `config_request` asks for them, `config_set` refuses to state them — an address read in a mail must never become where the token and the messages go). `RemoteMessagesHost` speaks every call as JSON over HTTP: `Wire` has the routes, the headers, the limits and `WIRE_VERSION`; the calls are derived from the interface's abstract methods, so a call added to `MessagesHost` is on the wire on both ends; an attachment crosses as the bytes of a GET, its size asked before it is read, never as base64; the token goes in the `Authorization` header and is in no URL, log line or error.
    - **The satellite** is the other end — a Keep install whose root config key `role` is `imessage_satellite` (`common/Role.py`), made by `install.sh` with `KEEP_ROLE=imessage_satellite`: `floxy_ai_server._launch` then builds no `Assistant` and imports nothing of `assistant`, `integrations` or `interfaces` (`tests/unit/test_satellite_role.py` boots one in a subprocess to prove it), only `Satellite.build()` — the local host behind `SatelliteService` (a Quart app that answers nothing without the token, compared in constant time before the route is looked up) on `imessage_satellite.port`, bound to 127.0.0.1 unless `imessage_satellite.lan_access`. It has no nightly clock: every request names the server's branch and commit, and `ServerFollower` brings the install there through `common/Updater` (`update`, then `relaunch("start.sh")`) — only `release` and `master`, one update at a time on its own thread, a target that failed left alone for an hour; its health answer carries its version and the last update failure (`HostHealth.satellite`). `Satellite.stay_up` does at each start what the server gets from its assistant: `Watchdog.sync` on the `watchdog` key, and the login item (`Autostart.enable`) once, at the first start that can read Messages. `role` and `imessage_satellite.*` are denied to `config_set` / `config_request` (`ConfigStore.DENIED_PREFIXES`).
    - **The typing dots** are made by setting the compose field of Messages through Accessibility, in a window of the chat's own that `TypingIndicator` opens from the chat's sidebar row and closes again; the host keeps them alive and ends them by itself after `TIMEOUT_SECONDS`, so a turn is one `on` and one `off` on the wire. Dots in the wrong chat are the one failure not allowed: the row must be the only visible one with the chat's exact name (`Contacts.sidebar_name` — a name two chats could answer to is none), the window must be new and titled with it before every write, every tree walk is capped, and anything else ends in no dots, a cleared field and a closed window. The gates (Accessibility granted, the session on the console, the screen unlocked, Messages open with its main window) are re-checked at every `on`; the one that is closed travels in `HostHealth.typing`. Every pyobjc call sits in `Accessibility.py`, loaded at first use and refused by the unit conftest — the tests run over `tests/unit/common/imessage/ax_fakes.py`, and nothing may call the real Accessibility API against Messages outside `tools/imessage_live_check/`.
    - **The scripts follow the role**: `scripts/role.sh`, sourced by `start.sh`, `stop.sh`, `autostart.sh` and `keep.sh`, reads `role` and `imessage_satellite.port` from `config.yaml` with awk, the way the Python side reads them (`tests/unit/test_role_sh.py` holds the two readings together), and what Python refuses to start on — an unknown role, an unusable port — stops `start.sh` before it kills anything. A satellite usually shares a Mac with its server (a second macOS user), so every process lookup of those scripts goes through `own_listeners` / `own_processes`: the caller's own processes only, the ports of the install's own role only, and a match by name only for a process running from this checkout — never a bare `pgrep -f` or `lsof -i` in a script. For a satellite `start.sh` waits for its port and ends with a verdict on the terminal instead of the console (it has none; `Updater.relaunch` runs the same script with the pid). The script tests stage a process table with users and checkouts (`tests/unit/process_table.py`).
- `src/assistant/` — the assistant proper; it never imports `interfaces.*`
  - `Assistant.py` (facade: `stream(RequestContext) -> AssistantEvents`, handler pool), 
  - `commands/` (Commands — the core `cmd_*` commands and the dispatch; `CommandSet.py` — the base of the commands an integration folder brings, discovered and hosted by `Commands`; CommandProcessor; `Toolkits.py` — the command references the LLM loads on demand; `formats/` — the call syntaxes the LLM writes a command in, `keep` (Keep's own text), `qwen` (the `<tool_call>` block), `dsml` (DeepSeek's block) and `invoke` (Claude's `<function_calls>` block, by name only), each a `CommandFormat` the processor parses through and `PromptManager` renders the prompt's declarations and examples in — one per process, `CommandFormats.active()`, chosen by the smart tier's LLM class or the tier's `command_format:` config key; `RescueFormat.py` wraps the processor's copy of any format but `invoke` so a call Opus writes as an `<invoke>` block anyway is read, logged at NOTE and stored in the active syntax — `format.rewritten` in `DialogManager`, before the turn joins the history. The texts follow one convention (`docs/plans/FORMAT_NEUTRAL_TEXTS_PLAN.md`): a mention is the bare backticked name, `name()` is only the declaration of a parameterless command, the prose states facts about values and never how to type them, and prose only Keep's syntax needs stands between `<!-- keep -->` and `<!-- /keep -->` lines — `tests/unit/assistant/commands/formats/test_texts_are_format_neutral.py` fails otherwise)
  - `prompt/` (PromptManager — assembles the prompt from `res/prompt.md` and the live sections; `PromptSectionGenerator.py` — the base of a section and the discovery of the ones integration folders bring; `sections/` — the core's own generators; MessageHistory)
  - `agents/CodeQualityMaintainer.py` runs the Claude Code CLI on the repo: the nightly run (test generation — the tests the last days' commits lack, a tests-only PR merged on the spot, a bug a new test exposes pinned as a strict xfail and listed, never fixed there —, bug analysis of the day's commits, log triage — the problems the last 24 hours of `log/` show, minus those a commit fixed since —, a PR fixing the Critical and Major items of the three fix lists, test health — flaky and slow tests —, auto-merge, and last the open pins — every strict xfail of the suite, numbered, with its fix complexity, the whole list every night until the maintainer fixes the bug or marks the test `@pytest.mark.wontfix`, a non-strict xfail the conftest maps and only a human adds; a night without commits keeps only the log triage, the fix, the auto-merge and the pins), the nightly update before it (pull the target branch, restart if the code changed; a process booting inside the hour takes the night over) and the commands of the `developer` toolkit — `self_analysis`, `improve_integration` (a PR confined to one integration folder, or a new one; the PR's files are checked after the run and a straying PR is closed), `report_issue` (a GitHub issue investigated and written for the developers), `try_pull_request` / `pr_feedback` (`agents/PullRequestTrial.py` — the server runs one of its own open PRs as `trial/pr-<N>` until the next `restart_server` or nightly update; the branch is the whole state, a PR job during a trial pushes to the tried PR, `start.sh` falls back to the normal branch when the PR's server never comes up) — and `self_improvement` (a PR anywhere), the one command of the `self_improvement` toolkit, offered only while the root config key `is_developer` is true (`Toolkit.requires`).
  - long-term memory, all under `memory/` (gitignored): `MemoryFile.py` — the bank `memory/bank.md` (people, preferences, contacts, rules; whole in every prompt; `remember` appends, the curator rewrites it; the prompt shows the curated part at the top and the notes appended since in a block of their own lower down — `MemoryFile.split`, so a `remember` never changes the cached top of the prompt); `WatchList.py` — `memory/watchlist/<slug>.md`, one file per tracked item (`save_watch` / `checked_watch` / `delete_watch` are the `watch` toolkit's, `res/toolkits/watch.md` — the prompt documents `load_watch` alone, which brings the toolkit along; the prompt carries a one-line-per-item index, `load_watch` returns an item whole, and the autonomous check gets the due items in full in its instruction, with the toolkit under them); `Procedures.py` — `memory/procedures/<slug>.md`, the how-tos (`load_procedure` loads one, `save_procedure` / `delete_procedure` write; the curator alone appends dated notes with `procedure_note`; the prompt carries only a code-built index). `memory/` holds only what is live: replaced, deleted and closed versions are copied to `log/memory/{watchlist,procedures}/`, a debugging trail the LLM never reads.
  - `agents/LongTermMemoryCurator.py` — the nightly run: rewrites the bank, then a `<commands>` block (`save_watch`, `delete_watch`, `save_procedure` for new names, `procedure_note`) tidies the watch list and may add (never replace or delete) procedures; a curated bank more than a tenth over `MEMORY_WORD_LIMIT` then gets the condensing pass (`MemoryCondenser`, section by section) — after the curation, never before it, so what is saved is the condensed bank; `//memory_curate` runs a curation over given days, `//memory_condense` the condensing pass alone, `//memory_split` is the one-off move of an old one-file bank into the three parts.
  - self-setup: `list_integrations` / `config_help` / `config_request` / `restart_integration` let the assistant configure an integration itself; a folder whose accounts sign in brings the command for it (`microsoft_login`, `google_login` + `google_login_code`), listed in the `setup` toolkit. The setup also registers `Keep Autostart.app` (`common/Autostart.py` — an applet in the root running `autostart.sh`, built like the Desktop shortcut by `common/Applet.py`) as a login item with the shortcut, so the server starts at login — still through `start.sh` and Terminal, so the Mac's grants apply. The root config key `watchdog` (off by default; `common/Watchdog.py`, applied at boot and by `config_set`) adds a launchd agent of the user's account that opens the same applet every 5 minutes, so a server that died is relaunched between logins — launchd runs only `open`, never a shell in the checkout, since a checkout under ~/Documents would hang a launchd shell on a consent dialog (#127).
  - `integration/` — the framework for outside services: `Integration`, `IntegrationMonitor` (a `check_health` that raises `HealthWarning` — the integration works and has something to report — is shown as WARNING and never counted towards ERROR), `Feed` / `FeedRunner` (a feed that raises `FeedSourceAway` — its source out of reach, not broken: a satellite that sleeps or restarts — keeps its cursor and is asked again for as long as it takes, instead of failing after `FAILURE_LIMIT`), `bootstrap.py` (discovery), `SatelliteHints.py` (what a problem with the Mac that holds a Messages account means to the owner, for both consumers of a `MessagesHost`: on a satellite's Mac `show_full_disk_access_setting` and `restart_server` do not reach) and the contracts more than one integration implements (`HomeAutomation` with `CompositeHome` — several backends as one house, `HomeAutomation.active()` resolves to it —, `CameraProvider` — `fetch_camera_images` fans out over `CameraProvider.active()` —, `VehicleProvider` with `Fleet` — the cars of every brand as one list, the `car_*` commands and the car section resolve a `vehicle_id` through `VehicleProvider.fleet()` —, `TfaCodeHandler`, and under `mail/` and `calendar/` the account contracts with their composites: `MailAccount` / `ImapMailAccount` (presets `icloud`, `gmail`) / `MailSource` / `MailAccess` and `CalendarAccount` / `Event` / `CalendarSource` / `CalendarAccess`). An integration folder implements `MailSource` / `CalendarSource` and holds one account per entry of its `<section>.accounts` list (`Accounts.py`: unique names, the id prefix each mail account derives from its name, `AccountHealth` for the per-account verdicts behind `accounts_status()`); the commands, the prompt section, the memory curator, the nightly report and Arlo's 2FA mail polling go through `MailAccess.current()` / `CalendarAccess.current()`, which gather the accounts of every enabled and configured integration through the monitor — never through a folder by name.
- `src/integrations/` — one folder per outside service (`tesla/`, `icloud/`, `photos/`, `imessage/`, `whatsapp/`, `home_assistant/`, `imap/`, `caldav/`, `google/`, `microsoft/` — the mail and calendar accounts, …) holding everything that belongs to it; the core finds each piece by discovery, so adding an integration is adding a folder. **The file suffix is the role, and each side loads only its own** — everything but `*Tab.py` is the assistant's and never imports `interfaces.*`:
  - `<Name>Integration.py` — the `Integration` class (`assistant/integration/bootstrap.py` walks the folders); helper modules beside it.
  - `<Name>Commands.py` — its commands, a `CommandSet`: `cmd_<name>` / `describe_<name>` methods and the flags of its own commands (`MULTI_PARAM_COMMANDS`, `SENSITIVE_COMMANDS`, `PURE_COMMANDS`, `APPROVAL_NOTES`). It reaches the shared machinery through the host (`self.background_command`, `self.profile`, `self.context`) and a helper that stays in core through a public member of the host (`self.commands.integration_monitor()`, `describe_restart`, `first_argument`, `request_route`, `media_tag`, `AI_MESSAGE_SUFFIX`), never a private one. `Commands` hosts every discovered set whether its integration is enabled or not and refuses a command name with two owners; `Commands.command_names()` is the full list without an instance. A command stays in `Commands.py` when more than one integration is behind it (home, mail, calendar, cameras, cars) or none.
  - `<Name>SectionGenerator.py` — its prompt data, a `PromptSectionGenerator` declaring `key` and `heading` (its own heading line, hashes included, plus an optional `intro`): the section stands under that heading in the prompt's one `{INTEGRATION_DATA}` block (the template never names an integration), or, with `sections:` in `toolkit.md` instead of a heading, is carried by the toolkit (`load_toolkit` then sends the data). `PromptManager` runs the discovered generators beside the core's (`prompt/sections/`: memory, procedures, watch list, vault, home, cars, calendar) and refuses a key with two owners. **A folder's generator runs only while the folder's integration is on**: `PromptManager` listens to the `IntegrationMonitor` (`folder_enabled` / `contract_enabled`, notified after each registration batch and every restart) and builds, starts, stops and drops the gated generators as the verdict changes — a core generator joins the gate with `needs = <contract>` (home, cars, calendar). A generator never checks whether its integration is enabled or waits for it to be built.
  - `<Name>Tab.py` — its console sub-tab, an `IntegrationTab` naming its `integration` class; on the Integrations row while that integration is enabled.
  - `toolkit.md` — the LLM's command reference when the folder alone owns the toolkit: front matter (`name`, `summary`, `commands`, `sections` / `hidden` when set) over the text; `assistant/commands/Toolkits.py` globs for it. A toolkit is offered only while its integration is on (`Toolkit.offered_by`, the monitor's verdict; a core toolkit fronting a contract declares `needs=` — home, cars, email, calendar): off, it is out of the `{TOOLKITS}` index, `load_toolkit` and the preloads answer that the integration is off, and `Commands.run` refuses its commands with the same answer. A disabled integration is visible to the LLM nowhere but `list_integrations`.
  - `config.example.yaml` — its config section with the how-to comments, read into the one schema by `common/ConfigSchema.py` (see Config example above).
  - `requirements.txt` — pins of packages only this folder imports, included from the root file with `-r`.

  **Local integrations**: an install's own integration folders can live outside the checkout — `extra_integration_dirs` in `config.yaml` lists directories laid out like `src/integrations/` (`~/KeepLocal` holding `schoolsoft/`, `podcasts/` …). `common/IntegrationDirs.py` resolves them; `src/integrations/__init__.py` appends them to the package's `__path__`, so a local folder is `integrations.<folder>` to every module discovery, and the file globs (toolkits, config fragments — appended after the repository's —, `requirements.txt` — a requirements group of its own, checked and installed with the rest) go through `IntegrationDirs.files()`. A local folder named like an earlier one is shadowed, logged and left out. The unit conftest pins the key to `[]`, so no test sees them.

  A broken folder costs only itself: a file discovery cannot read or import is logged and left out (for the folders of this repository the unit suite fails instead). Core reaches a folder through a contract or discovery, not by name — `tests/unit/test_integration_folders.py` holds the allowlists of the by-name imports that are left (core → folder, folder → folder, the launcher); shrink them, never grow them.

  **Adding an integration**: a folder with `__init__.py`; `<Name>Integration.py` whose class docstring has three paragraphs — what it is in a few words (the atlas description and the "Description" column of the console's Catalog tab), what it does for the user in one to three sentences (shown with the integration selected there), then the notes for whoever reads the code, which no UI shows — and whose constant `name` is what `enabled_integrations` lists; whichever of the pieces above it needs; a `GROUPS` blurb in `docs/atlas/descriptions.py`; tests in `tests/unit/integrations/<folder>/`. A folder that keeps a signed-in session in the tree lists its files in `wipe_targets()` and ends the session on the account in `log_out()` (`//wipe` gathers both by discovery); one with a login command of its own names it in `login_hint`. What is part of the prompt has a pinned order — give a new toolkit its place in `Toolkits.INDEX_ORDER` and a new section its place in `PromptManager.SECTION_ORDER`.
- `src/interfaces/` — one package per way of reaching the assistant, each an `Interface` (`enabled(config)`, `start(assistant, app)`, `start_async()`, `stop()`); a disabled interface's module is never imported
  - `http/` — HttpInterface (Quart app, `/stream`, `/health`, hypercorn, signals), `routes.py`, `WireFormat.py`
  - `voice/` — VoiceInterface (whole VoiceToVoice stack: microphones, TTS, server audio, speaker id, iOS overlay), `VoiceTurnGate.py`
  - `chat/` — the chat contract every chat UI shares: `ChatTransport` ABC + `Draft` protocol (what a platform implements). A question only the owner answers (an approval, a secret) is asked in the transports' `owner_conversations()`; a turn from any other conversation has it asked there instead, and its own transport is told (`ChatTransport.asked_elsewhere`). `IncomingMessage.context` is what a transport that reads a whole chat puts in front of a turn, outside the sender's own words
  - `telegram/` — TelegramInterface + `TelegramTransport` (the bot, a thin `ChatTransport`: update parsing, `TelegramDraft` send/edit/flood handling, status line, approval keyboards)
  - `imessage/` — IMessageInterface + `IMessageTransport` (`interfaces.imessage.enabled`, off by default; listed by the monitor as "iMessage Chat"): the assistant as an iMessage contact of its own, a `ChatTransport` over a `MessagesHost` (`imessage_chat.host`) whose Messages account is the assistant's Apple ID — never the account the iMessage integration reads (both on one account: the chat refuses to start). It polls the host with a cursor of its own (`ChatState`, `cache/imessage_chat.json`: first start at "now", a message older than 15 minutes when read is context, not a turn) and serves the handles of `imessage_chat.owners` / `guests` / `bots` (`Allowlist`; an owner only over the iMessage service, anyone else the chat layer's guest; every entry of the three lists is typed by the user, like `host` and `host_token` — `config_set` refuses them). A 1:1 message is always a turn, a group message only by the group wake rule (`assistant/GroupWake.py`, shared with the Telegram feed: a reply to its own bubble, its name, the message right after its own); what does not become a turn is the backlog put in front of the next one as untrusted context. Nothing somebody outside the household chose reaches the model as written (`Untrusted.py`: a group's or a contact card's name only as a plain short name, a backlog line as one JSON string under a label the sender cannot choose), a group is a `post_message` destination only once an owner has written in it, and a guest's message that finds the assistant busy is held and offered again by the poll (`ChatTransport.turned_away`). Replies go to the chat's guid as plain bubbles (`PlainText`), capped per recipient per day; no prompt is ever shown there — `owner_conversations()` is empty, so approvals and secrets go to the console and Telegram and the chat gets one line. While a reply is worked on the chat sees the account typing (`imessage_chat.typing_dots`, on by default): `IMessageDraft` → `IMessageTransport.typing` → `host.typing`, told to the host on a thread of its own so no turn waits for a window to open on another Mac, and everything that is sent takes the dots away first (`_dots_off`, waited for `TYPING_OFF_WAIT` at most). The dots are best effort — a reply never depends on them — and a gate that keeps them away is a `HealthWarning` worded by `SatelliteHints.typing_dots`. The setup the assistant walks the owner through is the comment over `imessage_chat` in `config.example.yaml` (and over `imessage_satellite` for the satellite's install).
  - `console/` — ConsoleInterface + `ConsoleTransport` (the TUI's chat, a `ChatTransport` rendering into the Home tab's `ChatPane`) + entire TUI. The Integrations tab is `tabs/IntegrationsGroup.py`: its sub-tab row is not a list in code but follows the `IntegrationMonitor` — `tabs/IntegrationTabs.discover()` finds the `*Tab.py` of every integration folder, `tabs/CatalogTab.py` is always first and lists every integration, enabled or disabled, and after it a tab is built and shown while its integration is enabled (never for a disabled one). `tabs/AuthTab.py` is the sign-in board a login tab extends.

**Request Flow**: `interfaces/*` build a `RequestContext` → `Assistant.stream()` → `RequestProcessor` → `DialogManager` → LLM; commands run through `assistant/commands/CommandProcessor` and each round's result is persisted as a `command` history row (`MessageHistory.add_command_result`, rendered verbatim on the user side so the next turn still sees it); the events stream back to the interface (HTTP renders them with `WireFormat`; chat interfaces — Telegram, iMessage and the TUI Home chat — through `interfaces/chat/ChatSession` → the transport's `Draft`).

**Destinations**: a reply goes back to whoever asked; `post_message(destination, text)` says something somewhere else. `src/assistant/Destinations.py` is the registry the interfaces fill at start — the chat hub registers `owner` (the proactive route, `notify_chat_ids` on Telegram), the Telegram transport one destination per group on `allowed_chat_ids` (by title, the id as alias), the iMessage transport one per person on its lists and per group chat an owner has written in (by name, claimed only while nobody else answers to it — `claim` / `release`; a send answers with the reason when it fails), the voice interface the server's speakers under their room's name (`local_outposts.location`) — and the `{DESTINATIONS}` line of the prompt follows it. The autonomous check emits commands only and speaks through `post_message`; stray text after its commands is dropped.

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
  `extra_integration_dirs` to `[]`, `imessage.host` and `imessage_chat.host` to `local` (no test reaches an
  install's real satellite) and `interfaces.imessage.enabled` to false; patch any other config value a test
  depends on

**Tests not to write.** A test must assert something the author had to *derive*, never something copied from the
line it checks. Do not write a test that:
- asserts a constant, default or attribute equals its own definition (`assert DEFAULT_X == "xxx"`, `SUPPORTS_Y is True`)
- only checks a constructor stored its argument (`assert obj.llm is mock_llm`)
- mocks the collaborator and then asserts the mock handed back what it was told to (mock config in → same literals out)
- compares a user-facing string byte for byte (log lines, notices, help text) — assert the fact it carries, not the wording

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

**Why these flags?** `pytest.ini` runs the suite on 8 xdist workers (`--dist worksteal`; more workers measured slower on the 8P+2E cores) with 
`log_cli=true` and `-s`. `tests/unit/conftest.py` also orders the collection from the previous run's durations (`.pytest_cache/v/floxy/slow-tests`, so a fresh checkout's first run is a little slower) and freezes the heap at session 
end so the workers exit fast. It also keeps the garbage collector off during collection and freezes the collected heap, opens SQLite without fsync and stubs `sounddevice` (no PortAudio init per worker). **It also keeps the suite off the machine it runs on**, and whatever adds a way of reaching the real Mac adds its refusal there (`tests/unit/test_no_server_relaunch.py` holds each one): its `Popen` guard refuses what `common/Updater` does for real — a script handed the test process's pid, a git command that moves the checkout the suite runs in, a pip install, so a test of anything that updates fakes git, pip and the start script — an AppleScript naming Messages or a login item, every `launchctl` call and `osacompile` (the watchdog agent and the login applet of the developer's own account); `sqlite3.connect` refuses the Mac's own Messages and AddressBook databases, and the defaults of `common/imessage` point at a folder that does not exist; the pyobjc loader of `common/imessage/Accessibility.py` is replaced by a refusal; the iMessage chat's state file is a scratch path. The wall time is ~7 s, of which ~3.5 s is every worker importing the tree at collection — that is the floor, so keep the run phase lean: no real sleeps or timeouts above ~50 ms in a test, and nothing that touches the real machine (Photos library, audio). 

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
- `src/assistant/integration/mail/MailAccess.py`, `calendar/CalendarAccess.py` - The one mailbox and the one calendar the commands, the prompt section and the agents use, gathered per call from the accounts of every enabled integration (`Accounts.py` names the accounts and derives the id prefixes; `src/common/ConfigMigrations.py` moved the old one-account sections into the lists)
- `src/interfaces/Interface.py` - The interface contract; implementations in `src/interfaces/{http,voice,telegram,console,imessage}`
- `src/interfaces/chat/ChatTransport.py` - What a chat platform implements; `ChatSession.py` is the shared reply/intake logic; `ChatHub.py` the Notifier
- `src/assistant/GroupWake.py` - The group wake rule (a reply to the assistant's bubble, its name, the message right after its own; the last-sender bookkeeping, the hourly brake on other assistants): one piece the Telegram feed and the iMessage chat transport both decide by
- `src/common/imessage/MessagesHost.py` - The one interface to a Mac's Messages account and the values that cross it; `LocalMessagesHost.py` in-process, `RemoteMessagesHost.py` over HTTP to a satellite (`Satellite.py`, `SatelliteService.py`); `Hosts.py` builds the one a config section names. Design and open points: `docs/plans/IMESSAGE_HOST_PLAN.md`
- `src/common/Updater.py` - Pull, requirements sync and relaunch, for the server (through `CodeQualityMaintainer`) and for a satellite
- `src/interfaces/console/ConsoleTransport.py` - The TUI as a chat transport; `widgets/ChatPane.py` is the Codex-style pane it draws into; `HistoryFeed.py` puts the dialog history in the same pane
- `config.yaml` - API keys and configuration
- `requirements.txt` - Python dependencies (it includes each integration folder's own `requirements.txt`)
- `log/floxy.out` - Log file from latest run on the server GREAT for DEBUGGING!
- `res/prompt.md` - The prompt template send to the AI; the command references are the toolkits — `res/toolkits/<name>.md` for the ones more than one integration is behind (or none), `src/integrations/<folder>/toolkit.md` for the one an integration owns (shared by both prompts) — only the core commands are in the template, and the integrations' data stands in its one `{INTEGRATION_DATA}` block (each section brings its own heading), never under a heading of the template's own
- `testsuite_prompt.md` - (private, present only when linked) the prompt the prompt test suite runs against. It is built from `res/prompt.md` by the private tree's `testsuite_prompt/build.py` — the template's text with the household's baked data in place of the live sections, `{COMMAND_SYSTEM}` and the other start-time placeholders kept — so it is never edited by hand: after a change to `prompt.md`, run the builder (a private unit test fails until it is run)

## Architecture Atlas

`docs/atlas/` builds the Keep Atlas, the zoomable architecture map the readme links to: `python3 docs/atlas/build.py`
(stdlib only) writes `docs/atlas/out/index.html`, and `.github/workflows/pages.yml` publishes it to GitHub Pages
at https://keep-ai.dev/atlas/ on every push to master. A module's description is the first sentence of its docstring (its main class's when it has one), so a
new module needs a docstring that opens with one plain sentence. `docs/atlas/descriptions.py` holds the package blurbs,
overrides for missing or overlong docstrings, and the "Follow a request" tour; **add a package blurb to `GROUPS` when you
create a package, and update `TOUR` when a module it names moves**. `tests/unit/test_architecture_atlas.py` fails on
either omission.

## Public repository

The repository is public: every commit, branch, tag, issue and PR body is public the moment it reaches GitHub — not
only the tree. The Pages site is keep-ai.dev (the domain is on Cloudflare, DNS-only records pointing at GitHub
Pages, the old maciekd.github.io address redirects there); it holds the Keep Atlas alone and its root redirects to
the repository.

## Dependencies

### Python
- **Requirements files**: `requirements.txt`, `requirements-voice.txt` (the voice stack), and `src/integrations/<name>/requirements.txt` for what only one integration needs — included from `requirements.txt` by `-r`, so installing the root file installs them all (`tests/unit/common/test_requirements_checker.py` fails for a folder file the root does not include)
- **Virtual environment**: `.venv` in the repository root
- **IMPORTANT**: Always activate venv before running Python: `source .venv/bin/activate`

### Testing Python Code
When testing code that imports `assistant`, `common` or `interfaces`:
1. Scripts importing these modules need `config.yaml` to exist
2. Always activate venv: `source .venv/bin/activate`
3. `ConfigManager` resolves `config.yaml` as `$FLOXY_CONFIG` if set, else `common.Paths.CONFIG_PATH` (= `config.yaml` in the root)

### System
- Python 3.12
- brew: mpv, portaudio
- Porcupine wake word models
- Spacy model: `en_core_web_sm`

## Database
- SQLite database: `floxy.db`
