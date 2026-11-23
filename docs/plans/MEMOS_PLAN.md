# Memos — long-term memory that is fetched

## Why

The memory bank is in every prompt and has a word limit. On a household that uses the assistant a lot it sits
above the limit night after night: the curation adds more than it removes, and the condenser only rewords — it
has no way to take a block out without losing it. A large part of such a bank is lookup material: a trip's
booking references, a wine list, test results, a directory of tradespeople. It is needed when its subject comes
up and is dead weight in every other turn.

A **memo** is that material, moved out: a named free text under `memory/memos/`, one index line in the prompt,
loaded with `load_memo` when needed. The bank keeps what acts on every turn — who is who, the rules, the
sensitivities, the preferences that shape an answer nobody asked about.

## A memo beside a procedure

Same mechanics, opposite rule for the curator:

| | Procedure | Memo |
|---|---|---|
| Holds | how to do something — steps that worked | what to know about something — facts |
| Shape | name, `when`, text | name, `when`, text |
| In the prompt | an index line | an index line |
| Loaded with | `load_procedure` | `load_memo` |
| The assistant | saves, replaces, deletes | saves, replaces, deletes |
| The curator | adds a new one, appends a dated note; never replaces or deletes | saves, replaces and deletes freely |

A procedure is protected because every step and literal matters. A memo is kept current: a new test round is
written into it, a trip's memo is deleted when the trip is over.

## The code

`assistant/Procedures.py` is a module of functions today. It becomes a class over a shared base:

- `assistant/MemoryDocs.py` — `class MemoryDocs`, the abstract base: a folder of named markdown files with a
  `When:` line. Everything both kinds share lives here as classmethods: `slug`, `names`, `read`, `when`, `save`,
  `delete`, `load`, `render_index`, `version`, `run`, the copy of a replaced or deleted version to the log, the
  vault redaction, the size caps. A subclass states its folder, its log folder, its noun (for the messages), its
  command names, its header prefix and its size cap.
- `assistant/Procedures.py` — `class Procedures(MemoryDocs)`: adds `note` (`procedure_note`) and keeps
  `allow_replace=False` for the curator.
- `assistant/Memos.py` — `class Memos(MemoryDocs)`: nothing of its own beyond the configuration. The size cap
  is the base's, the same for both (a memo that outgrows it is two memos).

Call sites keep reading `Procedures.names()`, `Procedures.load(name)` — they import the class instead of the
module.

Beside it, each with the procedures' counterpart as the pattern:

- `Commands`: `cmd_load_memo` / `cmd_save_memo` / `cmd_delete_memo`, their `describe_*`, `load_memo` among the
  pure commands and the loads (`Toolkits.LOADS`), and a loaded memo recognised as reference text by its header
  (so the history and the curation transcript treat it like a loaded procedure).
- `prompt/sections/MemosSectionGenerator.py` and `{MEMOS}` in `res/prompt.md`, with its place in
  `PromptManager.SECTION_ORDER`. The two generators share their one-second version watch through a small base.
- `res/prompt.md`: a short "Memos" passage beside "Procedures" — the three commands and the one distinction
  (facts to look up, not steps to follow).
- Console: `MemosTab` beside `ProceduresTab`.
- Telemetry (`memory_size` gains the memo count), the backup's description, `CLAUDE.md`, the atlas.

## The curator

This is where the bank gets shorter.

- The curator's prompt shows the memos **in full** (it shows the procedures as an index only): it cannot keep
  a memo current without reading it.
- One new rule, the twin of the how-to rule: lookup material — facts needed only when their subject comes up —
  belongs in a memo; `save_memo` it and leave at most a one-line pointer in the bank. What must act unasked
  (a rule, a sensitivity, a secrecy clause) stays in the bank even when the facts around it move.
- Its commands gain `save_memo` (new or replacing) and `delete_memo`.
- The safety net the procedures have applies: a refused `save_memo` keeps its text in the bank.
- A memo the period did not touch is not written at all — unlike the bank, it does not pass through the model
  every night.

The condenser stays as it is. With the lookup material gone the bank should fall under the condense trigger on
its own; if it does not, giving the condenser the same move is the next step, not part of this plan.

## Rolling it out

Nothing is migrated: an install starts with no memos and the curator moves material over the following nights.
On a large bank the first night is the big one, so on the maintainer's install it is run by hand
(`//memory_curate`) and the bank, the new memos and the journal's diff are read before the nightly run takes over.

## Tests

- `MemoryDocs` through both subclasses: save, replace with the log copy, delete, the caps, the index, `run`'s
  argument handling.
- What differs: the curator may replace and delete a memo and may not a procedure; `procedure_note` exists for
  procedures alone.
- The curator: `save_memo` / `delete_memo` in its command block, a refused save keeps the text in the bank, the
  memos' full text in its prompt.
- The prompt: the `{MEMOS}` index, the section order, the texts format-neutral.

## Not in this plan

- Renaming "procedure".
- Moving existing procedures that are really memos (a recipe collection, payment preferences at the top of a
  checkout how-to) — the assistant or the curator can do that later, one at a time.
- The condenser moving blocks.

## Open points

- **A cap on the memos' total.** A memo has the procedures' size cap. The curator's prompt carries every memo
  in full; a dozen at a hundred-odd words is nothing, a hundred is a cost. Start with a logged total, decide on
  a total cap when there is a number to look at.
- **Passive use.** A fact in a memo is used only when the model thinks to load it. The `when` line is what
  makes it do so — worth reading the first weeks' history for subjects that came up without their memo being
  loaded.
- **Index cost.** Every memo is a line in every prompt, so a block under roughly a hundred words is not worth
  moving. The curator's rule says so in a few words; whether it holds is to be seen in the first runs.
