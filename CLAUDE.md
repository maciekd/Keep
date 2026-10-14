# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Floxy AI is a home automation voice assistant: a Python server (`floxy_ai_server.py`) providing an intelligent AI
assistant that the user can speak to, with multiple LLM backends (GPT, Claude, Gemini, Ollama). This repository is the
server. Its clients are called "Outposts" — they act as "mics": local ones are microphones on the server machine,
remote ones (Android / iOS apps, not part of this repository) run on-device STT and send text to the server.

The system uses a predefined wake word, uses on-device and cloud speech-to-text, then sends the user query as text to 
the server. The server uses an LLM to process the query and can either play back the audio through local speakers or 
send the stream back to the client for playback. 

## Git Workflow

- **Only commit when explicitly asked** — never commit automatically after completing work.
- **Always use the `/commit` command** when committing changes. Never commit manually.
- **A push to master succeeds.** The remote prints "Changes must be made through a pull request" on every
  push — it is a notice, not a rejection; check `git log origin/master` before concluding a push failed.
- **Never `git reset --hard`, `git checkout -- <file>` or `git stash` a tree with changes you did not make.**
  Other sessions work in this same checkout; their uncommitted edits are on disk and nowhere else.

## Releases

Development happens on `master`; installs track the `release` branch. `./release.sh` fast-forwards
`origin/release` to `origin/master` and tags the commit with its version (`vYYYY.MM.N`, see
`src/common/Version.py`) — that is the whole release. The readme's install command fetches
`install.sh` from `release` and the installer clones it. `restart_server` checks out the branch the
`developer_release` config key names (false → `release`, true → `master`; the assistant may flip it with
`config_set`) and pulls it, so an install picks a release up on its next restart and a machine switches
line by flipping the key and restarting. The development machine's `config.yaml` has it true. Between the
pull and the restart `RequirementsChecker.sync` runs pip when the pulled pins no longer match the venv,
so a release may bump a pin freely; a failed install leaves the old process running and says why. The
nightly update (`CodeQualityMaintainer._nightly_update`, `nightly_update` config key, on by default) does
the same checkout and pull at 03:00 — after the nightly code-quality run when that is on — and restarts
only when HEAD is no longer the commit the process booted on. Never
hard-code `master` in a git command — `CodeQualityMaintainer._current_branch()` is the branch to come
back to after a Claude Code run, `target_branch()` the one to pull.

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

**Layout**: the root holds only scripts (`floxy_ai_server.py`, `tui.py`, `*.sh`), config, data (`res/`, `cache/`, `log/`, `voice_id/`, `memory/`, `floxy.db`), `tests/` and `tools/` (standalone maintainer scripts the server never imports, one folder each — `tools/generate_html_report_from_logs/` renders a copied `log/` folder as a chat-style HTML transcript); the four source packages live under `src/` and are imported without a `src.` prefix (`from common.x import …`). `pytest.ini` adds `src` via `pythonpath`; `floxy_ai_server.py` and every external script insert `<root>/src` at the front of `sys.path`. `common/Paths.py` (`AISERVER_DIR` — the repository root, `REPO_DIR` — the same, `SRC_DIR`, `INTEGRATIONS_DIR`, `RES_DIR`, `CACHE_DIR`, `LOG_DIR`, `VOICE_ID_DIR`, `MEMORY_DIR`, `CONFIG_PATH`) is the one place that knows where the tree is — never do `Path(__file__).parents[N]` arithmetic in a package module.
- `floxy_ai_server.py` — thin launcher: `build(port, testsuite) -> Server` constructs the `Assistant` and starts every enabled interface; `main()` serves (only under `__main__`)
- `src/common/` — logging, ConfigManager (which runs `ConfigMigrations` — the one-off moves of config.yaml sections, such as the mail and calendar accounts — before the sections are read), Paths, DevelopmentConfig, FloxyDatabase, monitors, RequirementsChecker
- `src/assistant/` — the assistant proper; it never imports `interfaces.*`
  - `Assistant.py` (facade: `stream(RequestContext) -> AssistantEvents`, handler pool), 
  - `commands/` (Commands — the core `cmd_*` commands and the dispatch; `CommandSet.py` — the base of the commands an integration folder brings, discovered and hosted by `Commands`; CommandProcessor; `Toolkits.py` — the command references the LLM loads on demand)
  - `prompt/` (PromptManager — assembles the prompt from `res/prompt.md` and the live sections; `PromptSectionGenerator.py` — the base of a section and the discovery of the ones integration folders bring; `sections/` — the core's own generators; MessageHistory)
  - `agents/CodeQualityMaintainer.py` runs the Claude Code CLI on the repo: the nightly run (bug analysis of the day's commits, log triage — the problems the last 24 hours of `log/` show, minus those a commit fixed since —, a PR fixing the Critical and Major items of both fix lists, test generation, test health — flaky and slow tests —, auto-merge; a night without commits keeps only the log triage, the fix and the auto-merge), the nightly update after it (pull the target branch, restart if the code changed) and the commands of the `developer` toolkit — `self_analysis`, `improve_integration` (a PR confined to one integration folder, or a new one; the PR's files are checked after the run and a straying PR is closed), `report_issue` (a GitHub issue investigated and written for the developers), `try_pull_request` / `pr_feedback` (`agents/PullRequestTrial.py` — the server runs one of its own open PRs as `trial/pr-<N>` until the next `restart_server` or nightly update; the branch is the whole state, a PR job during a trial pushes to the tried PR, `start.sh` falls back to the normal branch when the PR's server never comes up) — and `self_improvement` (a PR anywhere), the one command of the `self_improvement` toolkit, offered only while the root config key `is_developer` is true (`Toolkit.requires`).
  - long-term memory, all under `memory/` (gitignored): `MemoryFile.py` — the bank `memory/bank.md` (people, preferences, contacts, rules; whole in every prompt; `remember` appends, the curator rewrites it; the prompt shows the curated part at the top and the notes appended since in a block of their own lower down — `MemoryFile.split`, so a `remember` never changes the cached top of the prompt); `WatchList.py` — `memory/watchlist/<slug>.md`, one file per tracked item (`save_watch` / `checked_watch` / `delete_watch`; the prompt carries a one-line-per-item index, `load_watch` returns an item whole, and the autonomous check gets the full list in its instruction); `Procedures.py` — `memory/procedures/<slug>.md`, the how-tos (`load_procedure` loads one, `save_procedure` / `delete_procedure` write; the curator alone appends dated notes with `procedure_note`; the prompt carries only a code-built index). `memory/` holds only what is live: replaced, deleted and closed versions are copied to `log/memory/{watchlist,procedures}/`, a debugging trail the LLM never reads.
  - `agents/LongTermMemoryCurator.py` — the nightly run: rewrites the bank, then a `<commands>` block (`save_watch`, `delete_watch`, `save_procedure` for new names, `procedure_note`) tidies the watch list and may add (never replace or delete) procedures; `//memory_split` is the one-off move of an old one-file bank into the three parts.
  - self-setup: `list_integrations` / `config_help` / `config_request` / `restart_integration` let the assistant configure an integration itself; a folder whose accounts sign in brings the command for it (`microsoft_login`, `google_login` + `google_login_code`), listed in the `setup` toolkit.
  - `integration/` — the framework for outside services: `Integration`, `IntegrationMonitor`, `Feed` / `FeedRunner`, `bootstrap.py` (discovery) and the contracts more than one integration implements (`HomeAutomation` with `CompositeHome` — several backends as one house, `HomeAutomation.active()` resolves to it —, `CameraProvider` — `fetch_camera_images` fans out over `CameraProvider.active()` —, `VehicleProvider` with `Fleet` — the cars of every brand as one list, the `car_*` commands and the car section resolve a `vehicle_id` through `VehicleProvider.fleet()` —, `TfaCodeHandler`, and under `mail/` and `calendar/` the account contracts with their composites: `MailAccount` / `ImapMailAccount` (presets `icloud`, `gmail`) / `MailSource` / `MailAccess` and `CalendarAccount` / `Event` / `CalendarSource` / `CalendarAccess`). An integration folder implements `MailSource` / `CalendarSource` and holds one account per entry of its `<section>.accounts` list (`Accounts.py`: unique names, the id prefix each mail account derives from its name, `AccountHealth` for the per-account verdicts behind `accounts_status()`); the commands, the prompt section, the memory curator, the nightly report and Arlo's 2FA mail polling go through `MailAccess.current()` / `CalendarAccess.current()`, which gather the accounts of every enabled and configured integration through the monitor — never through a folder by name.
- `src/integrations/` — one folder per outside service (`tesla/`, `icloud/`, `photos/`, `imessage/`, `whatsapp/`, `home_assistant/`, `imap/`, `caldav/`, `google/`, `microsoft/` — the mail and calendar accounts, …) holding everything that belongs to it; the core finds each piece by discovery, so adding an integration is adding a folder. **The file suffix is the role, and each side loads only its own** — everything but `*Tab.py` is the assistant's and never imports `interfaces.*`:
  - `<Name>Integration.py` — the `Integration` class (`assistant/integration/bootstrap.py` walks the folders); helper modules beside it.
  - `<Name>Commands.py` — its commands, a `CommandSet`: `cmd_<name>` / `describe_<name>` methods and the flags of its own commands (`MULTI_PARAM_COMMANDS`, `SENSITIVE_COMMANDS`, `PURE_COMMANDS`, `APPROVAL_NOTES`). It reaches the shared machinery through the host (`self.background_command`, `self.profile`, `self.context`) and a helper that stays in core through a public member of the host (`self.commands.integration_monitor()`, `describe_restart`, `first_argument`, `request_route`, `media_tag`, `AI_MESSAGE_SUFFIX`), never a private one. `Commands` hosts every discovered set whether its integration is enabled or not and refuses a command name with two owners; `Commands.command_names()` is the full list without an instance. A command stays in `Commands.py` when more than one integration is behind it (home, mail, calendar, cameras, cars) or none.
  - `<Name>SectionGenerator.py` — its prompt data, a `PromptSectionGenerator` declaring `key` and `heading` (its own heading line, hashes included, plus an optional `intro`): the section stands under that heading in the prompt's one `{INTEGRATION_DATA}` block (the template never names an integration), or, with `sections:` in `toolkit.md` instead of a heading, is carried by the toolkit (`load_toolkit` then sends the data). `PromptManager` runs the discovered generators beside the core's (`prompt/sections/`: memory, procedures, watch list, vault, home, cars, calendar) and refuses a key with two owners. **A folder's generator runs only while the folder's integration is on**: `PromptManager` listens to the `IntegrationMonitor` (`folder_enabled` / `contract_enabled`, notified after each registration batch and every restart) and builds, starts, stops and drops the gated generators as the verdict changes — a core generator joins the gate with `needs = <contract>` (home, cars, calendar). A generator never checks whether its integration is enabled or waits for it to be built.
  - `<Name>Tab.py` — its console sub-tab, an `IntegrationTab` naming its `integration` class; on the Integrations row while that integration is enabled.
  - `toolkit.md` — the LLM's command reference when the folder alone owns the toolkit: front matter (`name`, `summary`, `commands`, `sections` / `hidden` when set) over the text; `assistant/commands/Toolkits.py` globs for it. A toolkit is offered only while its integration is on (`Toolkit.offered_by`, the monitor's verdict; a core toolkit fronting a contract declares `needs=` — home, cars, email, calendar): off, it is out of the `{TOOLKITS}` index, `load_toolkit` and the preloads answer that the integration is off, and `Commands.run` refuses its commands with the same answer. A disabled integration is visible to the LLM nowhere but `list_integrations`.
  - `config.example.yaml` — its config section with the how-to comments, read into the one schema by `common/ConfigSchema.py` (see Config example above).
  - `requirements.txt` — pins of packages only this folder imports, included from the root file with `-r`.

  A broken folder costs only itself: a file discovery cannot read or import is logged and left out (for the folders of this repository the unit suite fails instead). Core reaches a folder through a contract or discovery, not by name — `tests/unit/test_integration_folders.py` holds the allowlists of the by-name imports that are left (core → folder, folder → folder, the launcher); shrink them, never grow them.

  **Adding an integration**: a folder with `__init__.py`; `<Name>Integration.py` whose class docstring has three paragraphs — what it is in a few words (the atlas description and the "Description" column of the console's Catalog tab), what it does for the user in one to three sentences (shown with the integration selected there), then the notes for whoever reads the code, which no UI shows — and whose constant `name` is what `enabled_integrations` lists; whichever of the pieces above it needs; a `GROUPS` blurb in `docs/atlas/descriptions.py`; tests in `tests/unit/integrations/<folder>/`. A folder that keeps a signed-in session in the tree lists its files in `wipe_targets()` and ends the session on the account in `log_out()` (`//wipe` gathers both by discovery); one with a login command of its own names it in `login_hint`. What is part of the prompt has a pinned order — give a new toolkit its place in `Toolkits.INDEX_ORDER` and a new section its place in `PromptManager.SECTION_ORDER`.
- `src/interfaces/` — one package per way of reaching the assistant, each an `Interface` (`enabled(config)`, `start(assistant, app)`, `start_async()`, `stop()`); a disabled interface's module is never imported
  - `http/` — HttpInterface (Quart app, `/stream`, `/health`, hypercorn, signals), `routes.py`, `WireFormat.py`
  - `voice/` — VoiceInterface (whole VoiceToVoice stack: microphones, TTS, server audio, speaker id, iOS overlay), `VoiceTurnGate.py`
  - `chat/` — the chat contract every chat UI shares: `ChatTransport` ABC + `Draft` protocol (what a platform implements)
  - `telegram/` — TelegramInterface + `TelegramTransport` (the bot, a thin `ChatTransport`: update parsing, `TelegramDraft` send/edit/flood handling, status line, approval keyboards)
  - `console/` — ConsoleInterface + `ConsoleTransport` (the TUI's chat, a `ChatTransport` rendering into the Home tab's `ChatPane`) + entire TUI. The Integrations tab is `tabs/IntegrationsGroup.py`: its sub-tab row is not a list in code but follows the `IntegrationMonitor` — `tabs/IntegrationTabs.discover()` finds the `*Tab.py` of every integration folder, `tabs/CatalogTab.py` is always first and lists every integration, enabled or disabled, and after it a tab is built and shown while its integration is enabled (never for a disabled one). `tabs/AuthTab.py` is the sign-in board a login tab extends.

**Request Flow**: `interfaces/*` build a `RequestContext` → `Assistant.stream()` → `RequestProcessor` → `DialogManager` → LLM; commands run through `assistant/commands/CommandProcessor` and each round's result is persisted as a `command` history row (`MessageHistory.add_command_result`, rendered verbatim on the user side so the next turn still sees it); the events stream back to the interface (HTTP renders them with `WireFormat`; chat interfaces — Telegram and the TUI Home chat — through `interfaces/chat/ChatSession` → the transport's `Draft`).

**Destinations**: a reply goes back to whoever asked; `post_message(destination, text)` says something somewhere else. `src/assistant/Destinations.py` is the registry the interfaces fill at start — the chat hub registers `owner` (the proactive route, `notify_chat_ids` on Telegram), the Telegram transport one destination per group on `allowed_chat_ids` (by title, the id as alias), the voice interface a room (planned) — and the `{DESTINATIONS}` line of the prompt follows it. The autonomous check emits commands only and speaks through `post_message`; stray text after its commands is dropped. The plan: `docs/plans/POST_MESSAGE_PLAN.md`.

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
- a test must pass with `FLOXY_CONFIG=config.example.yaml`: patch the config values it depends on

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
end so the workers exit fast. It also keeps the garbage collector off during collection and freezes the collected heap, opens SQLite without fsync and stubs `sounddevice` (no PortAudio init per worker). The wall time is ~7 s, of which ~3.5 s is every worker importing the tree at collection — that is the floor, so keep the run phase lean: no real sleeps or timeouts above ~50 ms in a test, and nothing that touches the real machine (Photos library, audio). 

The suites outside `tests/unit` (`tests/prompt`, `tests/memory`, …) and `tests/unit/private` are **not part of this
repository**: they are built from a real household's data and are linked in (gitignored symlinks) on the maintainer's
machine only, together with `testsuite_prompt.md`, the prompt those suites run against. When they are present: they are
LLM/prompt-related and slow — only run them when changes affect LLM communication or prompts:
```bash
source .venv/bin/activate && python -m pytest tests/ --override-ini="log_cli=false" -q --tb=short 2>&1 | grep -E "^(FAILED|ERROR) |^=+ .*(passed|failed|error)"
```

**NOTE**: The prompt/integration tests spin up real server instances on fixed ports (8801+). Only one test suite run at a time per machine — concurrent runs will collide on ports and cause spurious "killed" failures.

## Key Files

- `floxy_ai_server.py` - Launcher (`build()` + `main()`)
- `src/assistant/Assistant.py` - The facade every interface calls
- `src/assistant/DialogManager.py` - Core dialog logic
- `src/assistant/RequestProcessor.py` - One request: prompt → LLM → commands → history
- `src/assistant/integration/Integration.py` - The integration contract; `IntegrationMonitor.py` registers and health-checks them; the implementations are the folders of `src/integrations/`
- `src/assistant/integration/mail/MailAccess.py`, `calendar/CalendarAccess.py` - The one mailbox and the one calendar the commands, the prompt section and the agents use, gathered per call from the accounts of every enabled integration (`Accounts.py` names the accounts and derives the id prefixes; `src/common/ConfigMigrations.py` moved the old one-account sections into the lists)
- `src/interfaces/Interface.py` - The interface contract; implementations in `src/interfaces/{http,voice,telegram,console}`
- `src/interfaces/chat/ChatTransport.py` - What a chat platform implements; `ChatSession.py` is the shared reply/intake logic; `ChatHub.py` the Notifier
- `src/interfaces/console/ConsoleTransport.py` - The TUI as a chat transport; `widgets/ChatPane.py` is the Codex-style pane it draws into; `HistoryFeed.py` puts the dialog history in the same pane
- `config.yaml` - API keys and configuration
- `requirements.txt` - Python dependencies (it includes each integration folder's own `requirements.txt`)
- `log/floxy.out` - Log file from latest run on the server GREAT for DEBUGGING!
- `res/prompt.md` - The prompt template send to the AI; the command references are the toolkits — `res/toolkits/<name>.md` for the ones more than one integration is behind (or none), `src/integrations/<folder>/toolkit.md` for the one an integration owns (shared by both prompts) — only the core commands are in the template, and the integrations' data stands in its one `{INTEGRATION_DATA}` block (each section brings its own heading), never under a heading of the template's own
- `testsuite_prompt.md` - (private, present only when linked) the prompt the prompt test suite runs against. When it is present, MIRROR any change you make to prompt.md into it (its `# COMMAND SYSTEM` block must match byte for byte — `tests/unit/private/test_testsuite_prompt_mirror.py` checks)

## Architecture Atlas

`docs/atlas/` builds the Keep Atlas, the zoomable architecture map the readme links to: `python3 docs/atlas/build.py`
(stdlib only) writes `docs/atlas/out/index.html`, and `.github/workflows/pages.yml` publishes it to GitHub Pages
(under `atlas/`) on every push to master. A module's description is the first sentence of its docstring (its main class's when it has one), so a
new module needs a docstring that opens with one plain sentence. `docs/atlas/descriptions.py` holds the package blurbs,
overrides for missing or overlong docstrings, and the "Follow a request" tour; **add a package blurb to `GROUPS` when you
create a package, and update `TOUR` when a module it names moves**. `tests/unit/test_architecture_atlas.py` fails on
either omission.

## Public readme page

The repository is private; the readme is not. `docs/site/build.py` (stdlib only, renders through GitHub's Markdown API)
writes `docs/site/out/index.html` and the same workflow publishes it as the root of the Pages site,
https://maciekd.github.io/Keep/, with `docs/images/` beside it. Nothing else of the tree is published, so **the readme
may only show or link local files under `docs/images/`** — the build and `tests/unit/test_readme_site.py` fail otherwise.
Everything in `readme.md` and `docs/images/` is public the moment it reaches master. `docs/site/preview.sh` builds the
page locally and opens it in the browser — the way to look at a readme change before pushing it.

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
