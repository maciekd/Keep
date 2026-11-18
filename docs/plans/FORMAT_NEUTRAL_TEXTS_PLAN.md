# Format-neutral texts — the prose of the prompt and the toolkits stops assuming Keep's syntax

Follows the command formats work (the `formats/` package), which made the *calls* in the texts render in the active
format (the rewriter). This one makes the *prose around them* true in every format, closes the
rendering defects the rewriter still has, and routes the two texts that bypass it through it.

## Why

An audit on 2026-10-21 (every text rendered in the invoke and qwen formats and read) found three
kinds of inconsistency:

1. **Keep-only prose the rewriter cannot touch.** "Give it as a triple-quoted block", "wrap each
   argument in quotes so commas are preserved", "quoted", "the command parser closes on the first
   balanced `)`", "without the fifth argument", `"3"` spec spellings in quotes. In a block format a
   value is raw text and has no position; a model that obeys passes literal quote characters.
2. **Rendering defects of the rewriter.** A parameterless command written `name()` is read as an
   example and becomes a whole call block in the middle of a sentence — 45 places in 12 texts,
   declarations included ("deliver the most recent `<function_calls>…</function_calls>` to the
   user"). An example may contradict its own schema (`latest_email("@SaneLater", "unread")`
   against `unread_only: bool`). An empty string standing in for a skipped optional parameter
   (`save_watch("festival-tickets", """...""", "", "1d")`) renders as an empty named parameter.
3. **Texts that never reach the rewriter.** The autonomous check instruction
   (`AutonomousActionManager.AUTONOMOUS_ACTION_PROMPT`) writes Keep pseudo-calls and appends the
   watch toolkit through `Toolkits.render(toolkit)` with no substitute, while `load_toolkit`, the
   preloads and the debug commands substitute. `CommandProcessor` sends "Command 'name(params)'
   returned an error / was denied / timed out" and "the reply ended before the parenthesis opened
   by name( was closed" back to the model in every format.

The memory curator has its own `<commands>` block, parsed by its own parser, in Keep's syntax by
design: out of scope, and it stays Keep whichever format the chat tier runs.
`sonos_enqueue_and_play` has no declaration because it is in `Toolkits.UNDOCUMENTED`: deliberate.

## Scope

- The rewriter and the named-parameter formats (`src/assistant/commands/formats/`).
- The prose of `res/prompt.md`, `res/toolkits/*.md`, `src/integrations/*/toolkit.md`.
- The autonomous instruction and the processor's messages to the model.
- Unit tests that keep it so.
- Not in scope: the curator's command block, the Keep parser, the private prompt suites' content
  (they are re-run, not edited), the TUI.

## Design

### One convention for the texts

- **A mention is the bare backticked name**: `` `browse_close` ``, `` `load_toolkit` ``. Every
  format's instructions already say a bare name never runs (Keep: "leave out the parentheses").
- **`name()` is written for one thing only: the declaration of a parameterless command**, on its
  own line like every declaration (`` `browse_close()` — shut down the browser… ``). The rewriter
  renders it with `render_declaration` — the `<function>` line in invoke, the schema line in qwen
  and dsml, as written in Keep — never as an example block. `name()` of a command that has
  parameters is a lint failure (below); `render_mention` and its tests go, the convention makes
  it unreachable.
- **The syntax lives in the format's instructions and in the examples; the prose states facts
  about values.** Not "give the note as a triple-quoted block" but "the note may run to several
  lines" — or nothing, since Keep's own instructions already say how a multi-line value is
  written and the rendered example shows it. Not "pass `unread_only` as the second argument" but
  "pass `unread_only`". Not "wrap each argument in quotes so commas are preserved" at all.
- **Prose only one format needs goes in a gated block**: `<!-- keep -->` … `<!-- /keep -->` on
  lines of their own. The rewriter keeps the content for the Keep format and drops the block
  (markers, content and one trailing newline) for every other. Today one passage needs it: the
  `)` caveat of `browse_eval`, which is a fact about the Keep parser's raw-payload reading. (The
  alternative — a sentence in `KeepFormat.INSTRUCTIONS` naming `browse_eval` — ties the format to
  a toolkit's command; rejected.) The DSML format already documents its own `</parameter>` escape
  in its instructions, which is where a format's own caveat belongs.
- **Examples match their declaration**: a `bool` parameter is written `true`/`false`, an `int` or
  `float` a bare number, a variadic a comma list. `latest_email("@SaneLater", true)` — the handler
  already accepts `true` (`_UNREAD_TOKENS`).

### Named formats drop a value that equals the declared default

`save_watch("festival-tickets", """...""", "", "1d")` is Keep's only way to skip `due`; in a
named format the idiom is to leave the parameter out. `render_example` of the named formats
omits a value equal to the declared default (an empty string for `= ""`, `false` for `= false`).
The three `render_example` loops (invoke, qwen, dsml) are the same loop with a different call
renderer: lift it into `NamedParameterFormat.render_example`, with one hook per subclass for the
block text (`render_call` of the module; DSML's adds the `string` flag). Keep's rendering stays
the identity.

### The texts that bypass the rewriter

- `AutonomousActionManager._action_prompt` (the `AUTONOMOUS_ACTION_PROMPT.format(...)` call):
  the assembled instruction goes through `self.prompt_manager.render_template` so a canonical
  example in it renders, and the appended watch toolkit is rendered with the same substitute the
  preloader uses (`Toolkits.render(toolkit, prompt_manager.render_template)`). The prose follows
  the convention: `` `checked_watch` `` when nothing changed, `` `save_watch` `` with the new
  state, `` `load_watch` ``, `` `post_message` `` — bare names; "Output only commands (using
  proper command syntax)" becomes "Output only commands".
- `CommandProcessor`: one label for the messages back to the model and the approval title's
  fallback, `name (params)` — `f"{name} ({params_display})"`, the bare name when there are no
  parameters — in place of the five `name(params)` f-strings. The unterminated report says what
  happened without a parenthesis: "`name` was not run: the reply ended inside the call before it
  was closed, so it could not be read. Nothing after that point in the reply reached the user
  either. Received: …". `executed_commands` keeps its `name(params)` strings — internal, and
  `RequestProcessor` tests them with `startswith("load_toolkit(")`.

### What guards it

`tests/unit/assistant/commands/formats/test_texts_are_format_neutral.py`, over
`Signatures.texts()` plus the autonomous instruction, each a derived assertion:

- no `name()` of a command whose declaration has parameters (a mention must be bare);
- no Keep-only phrase outside a `<!-- keep -->` block: `triple.quoted`, `in quotes`, `quoted`,
  `parenthes`, `the command parser`, `(first|second|third|fourth|fifth|last) argument`,
  `use this syntax exactly`;
- every example's values fit the declaration's types (`bool` → `true`/`false`, `int`/`float` →
  a number), through `Signatures.read_values` kinds;
- rendered in invoke, qwen and dsml: no call block for a parameterless command (its `name()`
  renders as a declaration), no gated block's markers or content, and the rendered autonomous
  instruction holds no Keep call of `save_watch`.

## Steps — one fresh agent per step

Each step is written for an agent that has read nothing but this file and the repo. The brief:
*"Read docs/plans/FORMAT_NEUTRAL_TEXTS_PLAN.md and do step N. Fill in its Status line when
done."* A step ends with the unit suite green (`CLAUDE.md` → Testing Strategy), the Status line
filled in, and no commit (the user commits with `/commit`). Step 2 is the bulk of the work and
is mostly editing; step 1 must land first because step 2's lint test depends on its rules.

### Step 1 — The rewriter's rules and the named formats' examples

Goal: the rewriter renders `name()` of a parameterless command as its declaration, strips
`<!-- keep -->` blocks for every format but Keep, and `render_mention` is gone; the named
formats share one `render_example` that drops default-valued values.

- `Rewriter.rewrite`: an empty-parens call of a command whose declaration has no parameters
  takes `fmt.render_declaration(declaration)`; an empty-parens call of a command with
  parameters is left as written (the lint of step 2 forbids it). Remove `render_mention` from
  `CommandFormat` and `KeepFormat`, and the tests that assert it
  (`test_rewriter.py::test_mentions_non_commands_and_unreadable_calls_stay_as_written`,
  `test_invoke_format.py::test_no_keep_shaped_call_survives_in_a_rendered_text`'s mention line).
- Gated blocks: `<!-- keep -->\n…\n<!-- /keep -->\n` — for the Keep format the markers go and the
  content stays; for any other the whole block goes. A marker without its pair is left as
  written and logged (`trace`), never guessed at. The texts have no such block yet; the test
  uses a fixture.
- `NamedParameterFormat.render_example(name, values, declaration)`: the loop now in
  `InvokeFormat`/`QwenFormat`/`DsmlFormat`, pairing values with the declared names, folding a
  variadic tail into one JSON list, and skipping a value equal to the parameter's default
  (compare `Value.value` with `Declaration` default decoded as `_default` does). Subclasses keep
  only the block renderer. `Signatures.Value` already carries `kind` and `value`.
- Tests: `test_rewriter.py` (declaration rendering of `browse_close()` in each format, gated
  block kept/stripped/unpaired), `test_named_parameter_format.py` or the three format tests
  (the default-valued value left out, in all three), and
  `test_invoke_format.py::test_real_texts_render_every_example_as_a_block_and_every_declaration_as_a_function_line`
  adjusted: a parameterless `name()` counts as a declaration.
- Logging: `trace` the count of gated blocks kept or stripped per text when there are any.

Status: DONE 2026-10-21, not committed. `Rewriter.rewrite`: an empty-parens call of a parameterless command renders as `render_declaration`; one of a parameterised command is left as written and traced ("a mention must be the bare name"); `_gate` resolves `<!-- keep -->` … `<!-- /keep -->` (whole lines, `KEEP_ONLY_OPEN` / `KEEP_ONLY_CLOSE`) before the calls are found — kept without markers for `fmt.name == "keep"`, dropped for every other, an unpaired marker left as written and traced. `render_mention` is gone from `CommandFormat` and `KeepFormat`. `NamedParameterFormat.render_example` is the one loop (names in order, variadic tail as a JSON list, a value equal to the parameter's default skipped via `_is_default`, `argN` past the names) and hands `(key, text, is_string)` triples to the subclass's `_render_block`; the three formats keep only that. Tests: `test_rewriter.py` (`TestKeepOnlyPassages`, the default-valued value handed over, `restart_server()` as a declaration; the Keep identity test now expects the markers gone), `test_invoke_format.py` (the default left out of a block, parameterless `name()` counted as a declaration).

### Step 2 — The text pass and the lint

Goal: every text follows the convention; the lint test passes and stays.

The pass, file by file (line numbers of 2026-10-21):

- `res/prompt.md`: 90–91 drop "Give it as a triple-quoted block:" and keep the example as the
  sentence's end ("…how to tell that it worked, e.g. `save_procedure(...)`"); 107–108 "Give
  `text` as a triple-quoted block when it has commas or several lines" → "`text` may run to
  several lines", example kept; 136 `` `load_toolkit()`, `load_procedure()`, `load_watch()` `` →
  bare names; 164 and 168 the parenthetical declarations → bare names ("`load_procedure` loads
  one"), the declarations stand once at 77/86.
- `res/toolkits/watch.md`: 9–11 "Give the note as a triple-quoted block" → "the note may run to
  several lines"; examples kept (step 1 renders the skipped `due` right).
- `res/toolkits/browser.md`: 1 `` `open_url()` `` → bare; 14 the `browse_act("3=query\n")` and
  `browse_key("Enter")` examples stay (canonical); 23–27 the spec bullets lose their quotes
  (`` `3` ``, `` `3=2026-05-04` ``, `` `3=STMNT shampoo\n` ``, `` `4=2026-05-02; 5=4; 6=19:00; 7` ``
  — values, not calls); 83–89 "The argument is forwarded VERBATIM — do NOT wrap it in quotes,
  brackets or backticks" → "`js` is the source as written", and the `)` / `String.fromCharCode(41)`
  caveat into a `<!-- keep -->` block; 35, 55, 80, 91, 95, 96, 99, 100, 107, 109, 116: the
  parameterless mentions bare, the declaration lines (55, 80, 91, 99) keep `name()`; the
  Stop-and-wait list at 116 all bare.
- `res/toolkits/email.md`: 8 and 15–16 "Pass `unread_only` as the second argument" → "Pass
  `unread_only`", example `latest_email("@SaneLater", true)`; 11 `` `latest_email()` `` → bare;
  25 keeps `list_folders()` (declaration); 30 "and quote both arguments" dropped; 38–39 "Wrap
  each argument in quotes so commas inside the quoted body are preserved as part of the body"
  dropped, "A fourth argument names the account" → "`account` names the account"; 45 bare.
- `res/toolkits/calendar.md`: 11 "(use this syntax exactly)" dropped; 15 "Without the fifth
  argument" → "Without `calendar`".
- `res/toolkits/cars.md` 3, `res/toolkits/home.md` 10, `src/integrations/husdata/toolkit.md` 8,
  `src/integrations/music/toolkit.md` 13, `res/toolkits/timers.md` 6: "(use this syntax
  exactly)" dropped. `home.md` 47/50, `timers.md` 18, `husdata` 10: the declaration keeps
  `name()`, the Stop-and-wait mention goes bare.
- `res/toolkits/developer.md` 33: "The first argument is the integration's name … The rest is
  the request" → "`integration` is the name as `list_integrations` shows it …; `request` is …";
  64, 76 `restart_server` bare except its declaration line.
- `res/toolkits/setup.md`: 76, 87, 95 ", quoted" dropped; 2, 15, 26, 49, 65, 113 and the
  Stop-and-wait list bare; the declaration lines (4, 45, 85, 91) keep `name()`.
- `res/toolkits/vault.md` 6 "Wrap each argument in quotes." dropped.
- `res/toolkits/voice.md` 27, 31: "respond with only the command `false_positive()` and nothing
  else" → "respond with `false_positive` alone".
- `src/integrations/homeclaw/toolkit.md` 26: "Add `"dry_run"` as the last argument" → "Pass
  `dry_run` as `dry_run`", example kept.
- `src/integrations/imessage/toolkit.md` 15–16, `src/integrations/telegram_feed/toolkit.md`
  13–14, `src/integrations/whatsapp/toolkit.md` 12–13: "Wrap each argument in quotes so commas …
  stay part of it" dropped, the examples kept after "e.g.".
- `src/integrations/icloud/toolkit.md` 20, 26, 43, 49; `src/integrations/dropbox/toolkit.md`
  21 (declaration, keeps `()`), 32; `src/integrations/weather/toolkit.md` 10 (declaration), 13;
  `src/integrations/music/toolkit.md` 37; `src/integrations/reminders/toolkit.md` 11
  (`list_reminders(list: str)` is a second declaration shape of a declared command — make it
  prose: "with `list` it reads one list"): mentions bare.
- Every other `name()` of a parameterised command in a Stop-and-wait list or in prose: bare.
  The lint finds them; do not hunt by eye.

Then the lint, `tests/unit/assistant/commands/formats/test_texts_are_format_neutral.py` as in
the design, and `tests/unit/assistant/commands/test_toolkits.py`
unchanged. The private `testsuite_prompt.md` is built from `res/prompt.md`: run the builder
(`CLAUDE.md` → Key Files) if the private tree is linked, else say so in the Status line.

Status: DONE 2026-10-21, not committed. Two mechanical passes first — every `` `name()` `` of a parameterised command to the bare name (16 texts), every mid-line `` `name()` `` of a parameterless command to the bare name (8 texts) — then the prose by hand as listed, plus: `false_positive`, `icloud_devices` and `icloud_unlocked` got a declaration line each (they had only mentions), `list_reminders` is declared once as `list_reminders(list: str = "")`, `show_photo(42, 57, 103)` for the `int` parameter, the browse_eval `)` caveat is the one `<!-- keep -->` passage. The lint is `test_texts_are_format_neutral.py` (bare mentions, no Keep-only wording outside a gated passage — `quoted` is matched only as `, quoted` / `quote both|each`, since the X toolkit says "quoted post" —, examples fit the declared types, no call block for a parameterless command in any block format, no marker in any rendering). The private `testsuite_prompt.md` was rebuilt with `KeepPrivate/testsuite_prompt/build.py`. The worked example of the browser toolkit now reads "You: `browse_extract`" for its parameterless steps.

### Step 3 — The texts that bypass the rewriter

Goal: the autonomous instruction and the processor's messages speak no syntax of their own.

- `AutonomousActionManager`: the prose of `AUTONOMOUS_ACTION_PROMPT` to bare names (lines 142,
  154: `checked_watch`, `save_watch`, `load_watch`, `post_message`), "Output only commands
  (using proper command syntax)" → "Output only commands"; `_watch_toolkit` renders with the
  prompt manager's `render_template` (static today, as `_action_prompt` is — both become instance methods, they need the manager); the assembled instruction passes through `render_template` last. Check
  `render_template` does not choke on the instruction's own braces (`.format` has run; a
  `{{secret:…}}` placeholder in a watch note must survive — test it).
- `CommandProcessor`: the label helper and the five messages (lines 118, 125, 139, 145, 149,
  223, 244–247), the unterminated report (595–597). `last_dispatched_display` has no reader outside the processor — label it the same way.
- Tests: `tests/unit/assistant/agents/test_autonomous_action_manager.py` (the instruction under
  the invoke format carries the watch toolkit with `<function>` lines and no `save_watch("`),
  `tests/unit/assistant/commands/test_command_processor_advanced.py` (the report names the command and
  not a parenthesis; the error message carries the name and the parameters).
- Logging: `trace` in `_watch_toolkit` already says the toolkit rides along; add the format name.

Status: DONE 2026-10-21, not committed. `_action_prompt(llm, prompt_manager=None)` keeps its static signature (a dozen tests call it bare): with a manager the formatted instruction goes through `render_template`, which renders the watch toolkit it carries and any call in it; `activity_check` passes `self.prompt_manager`. The prose names `checked_watch`, `save_watch`, `load_watch`, `post_message` bare and says "Output only commands". `CommandProcessor._label(name, params_display)` → `name (params)` / the bare name, used by the refused, test-mode, error, approval-title, denied, timed-out and unavailable messages and `last_dispatched_display`; the unterminated report says "the reply ended inside the call before it was closed". `executed_commands` keeps `name(params)` (RequestProcessor's `startswith("load_toolkit(")`). Tests: the autonomous instruction rendered under the invoke format carries `<function>` lines for `save_watch` / `checked_watch` and no Keep call shape (the mock prompt manager of `make_mock_prompt_manager` passes text through); the processor's report has no parenthesis talk outside its `Received:` tail, and `_label` is no call.

### Step 4 — Wrap-up

- `CLAUDE.md` → the `commands/` bullet: one sentence on the convention (a mention is the bare
  name, `name()` only declares a parameterless command, Keep-only prose in `<!-- keep -->`).
- The command formats plan (since removed) → its text-pass open point closed, pointing here.
- If the private suites are linked: `testsuite_prompt.md` rebuilt, `tests/prompt` run once (the
  wording changed for Claude too — only prose, the examples still carry Keep's syntax, but the
  suite is the check). If the local hosts are reachable (`reference_local_models_test_host`):
  one query each on Qwen and ds4 that loads the browser toolkit and calls `browse_close`, to see
  the rendered toolkit land and the parameterless call parse.
- The prompt cache of every tier invalidates once on deploy; nothing to do, worth knowing when
  reading the cache numbers that day.

Status: DONE 2026-10-21 but for the LLM runs, not committed. `CLAUDE.md` carries the convention. Unit suite: 10331 passed; the 10 failures left are not this plan's — 8 in `test_first_run.py::TestCurating` from the NOCOMMIT `return False` in `FirstRun.Pipeline` (the ds4 prefill experiment), 2 in `tests/unit/private/test_prompt_test_runner.py` where the private tree's `run.py` and that test are mid-edit by another session. Not run: `tests/prompt` (billed, slow) and the one-query check on the Qwen / ds4 hosts — the user's call.
