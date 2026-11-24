# Local AI from the installer

## Why

Keep runs on a local model (`local_backend`, `docs/local-ai.md`), but the installer knows only the cloud: it asks
for an Anthropic key, and somebody who wants everything on their own machine has to create an Anthropic account
first and move afterwards. The installer should offer the choice and, on a Mac that can carry it, set the local
model up itself.

Two constraints shape the plan:

- **Maintainable.** `install.sh` is 1,850 lines of bash, and the part that talks to a person and to an API is the
  part bash does worst. What is added here is Python, and the key and config steps move there with it.
- **One runner, set up plainly, and no recommendation that ages.** oMLX on every Mac, with its own defaults, in the
  one version and with the one model we tested. Whatever makes a model faster this month — Splash today, an
  engine oMLX can drive on an M3 or newer; next month a drafter or a newer quantization — is not installed, not named in a text of Keep's and not in the docs'
  table of supported setups: that table holds what we test. A user turns such a thing on in oMLX; Keep sees an
  OpenAI-compatible address either way.

## What the user sees

The step that is "Anthropic API key" today becomes **"AI model"**. It comes where it does now — after the Python
packages — because that is the first moment Python can run.

**A Mac that can run it** (see the gate) is asked one question:

```
Keep can think in the cloud or on this Mac.

  1  Claude (Anthropic)
     The most capable; secrets go in the Keep Vault, your requests go through Anthropic servers
  2  Local, automatic setup
     Qwen 3.8 27B on this Mac: an 18 GB download, 100% local AI
  3  Local, manual setup
     Set up your own local LLM model/server (advanced)

Choice [1]:
```

Choice 2 promises local AI, not privacy: telemetry stays on by default for every install and, for its first
seven days, sends each turn's query and reply in full. No text of the installer says "private" of a local install;
the summary's line on telemetry (what it sends, `telemetry.enabled: false`) is the same for both.

A choice is its name on one line and its description under it, dimmed and wrapped to the terminal's width: on
the 80 columns of a fresh Terminal window a description beside the name would break mid-line.

- **1** — the key flow as it is today.
- **2** — oMLX and the model are installed, the config is written with `local_backend: openai` and no Anthropic
  key. Anything that fails on the way says what failed and goes on as **1**: a broken local setup costs the user
  the local option, never the install.
- **3** — one line, "start on Claude and ask Keep to move you to your own local AI; it will set it up with you",
  then **1**.

**A Mac that cannot** gets one line saying which requirement is missing and that local AI is therefore not
offered, then **1**.

**Linux** gets no question: one line, "to run Keep on local AI, start on Claude and ask Keep to set it up", then
**1**.

**Unattended** (`ANTHROPIC_API_KEY` in the environment, or no terminal): no question, **1** as today.
`KEEP_AI=local` in the environment answers **2** without asking.

## The gate

Local is offered when all of these hold:

| | Requirement | Why |
|---|---|---|
| Chip | Apple Silicon | MLX |
| macOS | 15 or newer | the oMLX formula requires it (the installer's own floor stays 14) |
| Memory | 32 GB or more | the model and a long prompt take about 21 GB; 32 GB is what we support and what we say |
| Disk | 30 GB free | the model, oMLX (1.7 GB), its disk cache, on top of Keep's own 10 GB |

The line a failing Mac reads names the first one it misses, with its own number ("16 GB of memory; local AI needs
32 GB").

## What option 2 sets up

1. **Homebrew** — used when present, installed when not: option 2's text says so, and Homebrew's own installer
   runs (it asks for the password).
2. **oMLX, the version we tested** — `LocalRunner` names one release. `brew tap jundot/omlx
   https://github.com/jundot/omlx`, the tap checked out at that release's tag, `brew install jundot/omlx/omlx`,
   `brew pin` so a `brew upgrade` of the user's leaves it. Moving to a newer oMLX is a Keep release that changes
   the constant, after the proof below has passed on it. One already installed is kept as it is, whatever its
   version, and told apart in the log.
3. **The model** — `Jundot/Qwen3.8-27B-oQ4e-mtp` (17.5 GB: oMLX's own mixed-precision build, with vision and the
   MTP head) into `~/.omlx/models/`, with the venv's `hf download --revision <commit>`: the repository has been
   re-quantized in place before, so `LocalRunner` names the commit beside the oMLX version and the two move
   together. The download resumes, so an interrupted install continues where it stopped. A folder that already
   holds the model is kept.
4. **Settings** — `~/.omlx/settings.json`, as few keys as possible, since oMLX renames keys between releases: the
   memory ceiling on a Mac under 36 GB (below) and nothing else. An existing file is the user's: keys are added,
   never overwritten, and the port it names is the one Keep is pointed at.
5. **The service** — `brew services start jundot/omlx/omlx`: a launchd agent that starts oMLX at login and revives
   it. Keep's `start.sh` and `stop.sh` know nothing of it.
6. **The proof** — `GET /v1/models` lists the model, then a one-token completion, which loads the model and so
   proves it fits. Only then is the config written.
7. **`config.yaml`** — `local_backend: openai` and `openai_compatible.model: Qwen3.8-27B-oQ4e-mtp` (the name says qwen,
   which is what makes Keep write Qwen's call syntax); `base_url` only when the port is not 8000; no
   `api_keys.anthropic`.

The summary at the end says the model runs on this Mac, that oMLX starts at login, and how to go back
(`local_backend: none` and a key).

## The look

The install should be a pleasure to watch. Two things carry that.

**Progress that says how far and how long.** Wherever the total is known the line is a bar with a percentage,
the rate and the time left:

```
  Qwen 3.8 27B   ████████████████░░░░░░░░░░░░░░  53%   9.4 of 17.5 GB   41 MB/s   3 min left
```

- the model download — through `huggingface_hub` with a progress class of ours, so one bar for the whole model
  and not `hf download`'s bar per file;
- the Python packages — the bar is there today, counted in packages; it gains the time left;
- the clone of the repository — git reports its own percentage.

Where there is no total — Homebrew installing itself, `brew install` of oMLX, the model's first load — the line
is the spinner with what is happening now and the time it has taken. No bar is shown that would have to invent
its percentage.

**The console's colours.** The palette is `interfaces/console/Theme.py`, which `common/install` may not import
and bash cannot. The values move to `common/Palette.py` — plain RGB, importing nothing — and `Theme.py` builds
its colours from it. `install.sh` runs before there is a checkout, so it carries its own copy of the few it
uses, and a test holds the two together (as `test_role_sh.py` does for the role).

- **Accents only.** Green for done, yellow for a warning, red for a failure, cyan for the step in progress and
  the bar — which fades up from the accent's deep companion, as the console's meters do. The text itself stays
  in the terminal's own foreground: the console draws on black, and a fresh Mac's Terminal is black on white,
  where the palette's text greys would vanish.
- **True colour where the terminal has it** (`COLORTERM`), the nearest of the 256 colours otherwise — Terminal
  before macOS 26 has no true colour —, the plain ANSI colours of today under that, and none with `NO_COLOR` or
  without a terminal.
- One implementation per side: `Terminal.py` for Python, `init_ui` for bash, both reading the same names.

## The memory ceiling

oMLX (0.8.3) refuses to load a model, and aborts a request, above a ceiling it recomputes every second from what
the other apps hold: on 32 GB under its default tier roughly 29 GB minus wired and compressed memory minus three
quarters of the other apps' active memory. The build measured below, `mlx-community/Qwen3.8-27B-4bit`, is 15.7 GB and a 27k-token context brings it to about 20 GB.

Measured on the development Mac (M1 Max, 32 GB, many apps open, 2026-11-24):

| Setting | Ceiling | Load | 27k-token prompt |
|---|---|---|---|
| default (`balanced`) | 12.4 GB | refused | – |
| `aggressive` | 17–19 GB | ok | aborted after 6 minutes |
| fixed 22 GB | 22 GB | ok | ok, 75 tok/s prefill, 15 tok/s decode |

Projected, not measured, for a 32 GB Mac that runs macOS, Keep and a browser: the default ceiling is about
21.7 GB — it works with 1–2 GB to spare, and an IDE or a heavy browser session takes it under what a long prompt
needs. A turn that fails after minutes of prefill because the owner opened something is worse than a slower
browser, so:

- **under 36 GB**: a fixed ceiling of 22 GB (`memory.memory_guard_tier: custom`,
  `memory.memory_guard_custom_ceiling_gb: 22`). oMLX then never yields; macOS compresses and swaps the other apps.
- **36 GB and more**: oMLX's default, which there leaves more for the cache than 22 GB would.

The threshold and the number are values of `LocalRunner`, derived from the Mac's memory.

The installer's model, `Qwen3.8-27B-oQ4e-mtp`, on the same Mac under the same 22 GB ceiling (2026-11-24):

| | plain 4-bit | oQ4e-mtp |
|---|---|---|
| Loaded | 15.7 GB | 16.7 GB |
| Peak, 27k-token prompt | about 20 GB | about 21 GB — it fits, with 1 GB to spare |
| Prefill | 75–80 tok/s | 72 tok/s |
| Decode | 15–17 tok/s | 16–18 tok/s on a cached prompt, 13–14 right after a long prefill |

The MTP head brings nothing measurable on an M1 Max: 16.5 tok/s with it off, 15.7–18.3 with it on, and oMLX
parks it by itself when too few drafts are accepted. It stays at oMLX's default (on); the reason for this build is
its quantization, not its speed. A 1 GB margin is what a 32 GB Mac has — a build any larger does not fit.

## The code

Bash keeps what has to run before there is a Python: the checks, the system packages, the browser, Python itself,
the repository, the venv, the packages — and the launcher, the summary and the launch after. Everything between
is one call:

```
"$KEEP_DIR/.venv/bin/python" -m common.install
```

`src/common/install/` is a package of `common` so that the satellite role can use it: it imports nothing of
`assistant`, `integrations` or `interfaces`.

| Module | Holds |
|---|---|
| `__main__.py` | the steps in order, for the server and for the satellite; the exit status is the only thing bash reads |
| `Terminal.py` | asking on the terminal the installer was started from (it runs piped from curl), hidden input, the `ok` / `warn` / `note` lines, the progress bar and the spinner in the palette's colours, the install log |
| `AnthropicKey.py` | today's `obtain_api_key` and `verify_api_key`: the key, the workspace id, the credit wait — the HTTP call a method of its own, so the tests fake Anthropic's answers |
| `AiChoice.py` | the gate and the question |
| `LocalRunner.py` | oMLX: install, model, settings, service, proof. Every fact about oMLX is in this one file — replacing the runner is replacing it |
| `InstallConfig.py` | writing `config.yaml` (server and satellite) through `PrivateFile`, with a YAML library instead of a heredoc |

What crosses from bash is the environment the installer already has (`KEEP_DIR`, `BRANCH`, `ROLE`, `LOG`, the
telemetry id, `ANTHROPIC_API_KEY`, `KEEP_AI`). Nothing comes back but the exit status and the config file: bash
reads `local_backend` from it for the `install_finished` event (`ai=local` or `ai=cloud`), with the `config_value`
it already has. The machine facts for the gate go through `common/Platform.py`.

`scripts/local_omlx_start.sh` and its test are removed: the service owns port 8000. What its header knows about
memory moves to `docs/local-ai.md`.

## Tests

- The key flow's cases move from `tests/unit/test_install_sh.py` to plain pytest over `AnthropicKey` — each of
  Anthropic's answers a fake response, no staged curl.
- `LocalRunner` runs over fakes of `brew`, `hf` and the HTTP calls: already installed, a download that fails, a
  port in use, a model that does not load, an existing `settings.json` left intact.
- The unit conftest refuses the real things, as for everything else that reaches the machine: a `brew` command
  and a write under the real `~/.omlx`.
- `test_install_sh.py` keeps the bash side, with the Python step a stub that exits 0 or 1.
- The gate is tested on both systems through `Platform.current`.

## Order of work

1. **The move, no behaviour change.** `common/install/` with `Terminal`, `AnthropicKey`, `InstallConfig`; bash
   calls it; the key and config tests are rewritten. A release goes out on this alone, so a regression in the key
   flow is not mixed up with the new feature.
2. **Local.** `AiChoice`, `LocalRunner`, the config keys, the summary, the Linux line; `local_omlx_start.sh`
   removed; `docs/local-ai.md` (the supported row becomes oMLX with this model, Splash leaves the table), the
   readme and `CLAUDE.md` updated.
3. **Later, the same code from the assistant.** A command that runs `LocalRunner` on an install that started on
   Claude, so "ask Keep to move you to local" is this code and not advice. Not part of this plan's first release.

## To verify on a real Mac before release

- A fresh install with no Anthropic key: first run, the periodic check, vision and the nightly memory curation all
  work on the local model alone.
- A 32 GB Mac with nothing else installed: the model loads under oMLX's defaults and a first turn with Keep's real
  prompt answers.
- The install on a Mac that never had Homebrew.

## Open points

- **The ceiling on a clean 32 GB Mac.** The figures for a Mac that runs little besides Keep are a projection; the
  check on a real one (above) confirms or moves the 22 GB.
- **Installing a fixed version through Homebrew.** The tap is oMLX's own repository and carries its release tags
  (`v0.8.3`), so the tap at a tag is a `git checkout` in `brew --repo jundot/omlx`, with `HOMEBREW_NO_AUTO_UPDATE`
  for the install. Not yet run end to end; nor is it decided whether an existing install gets the pinned version
  on a later Keep release or only new installs do.
- **Install time of oMLX.** There is no bottle: the formula builds a venv with pip and compiles four packages
  from source, with Rust and Python 3.11 brought in by Homebrew first. Minutes — to be measured on a clean Mac,
  and the spinner's line says it is a build.

Settled since:

- **oMLX's authentication.** A server without a key accepts inference when it is bound to loopback
  (`omlx/server.py`, 0.8.3), which is what a fresh install is: step 4 sets no key and step 7 writes none. An
  existing `settings.json` that holds `auth.api_key` is the user's: `LocalRunner` reads it and writes it as
  `openai_compatible.api_key`.

## Status

Step 1 of the order of work is built (2026-11-24): `src/common/install/` with `Terminal`, `AnthropicKey`,
`InstallConfig`, and `install.sh` calling it through `configure`. Step 2 has not been started.
