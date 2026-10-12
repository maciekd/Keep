# Plan: `post_message` and destinations — the autonomous check emits commands only

Written 2026-10-12. Status: phases 1 and 2 built, in review.

## What we want

Two Keep instances (Maciek's and Erik's) sit in one Telegram group, "Maciek & Erik - Michelle test"
(`-5342761890`). The wish: the assistants share, in plain language, whatever they learnt that the
other household's Keep could use — a message board of tips. Trying to do that by hand on 2026-10-12
showed the gaps: a procedure posted in the group was clipped, and the mail fallback failed because a
5,000-character `send_email` approval card exceeded Telegram's message limit.

Working the idea down, the board turned out to be a special case of one general gap: **the
assistant has no way to say something to a chosen place.** A normal turn answers where it was
asked. The autonomous check (`AutonomousActionManager`) can only push its reply to the owner's
chats, through `notifier().send_message`. So the plan is not a board feature. It is:

1. **Destinations** — the places the assistant can deliver a message to: the owner's chats, a
   Telegram group the bot is in, a room's speakers.
2. **`post_message(destination, text)`** — one command that delivers there, usable in any turn.
3. **The autonomous check emits commands only.** Its reply is no longer "commands, then the
   Telegram text or NO": anything it wants said goes through `post_message`. Several messages to
   several places are several commands.

The board then needs no code: a daily watch item tells the assistant to post the day's lessons
with `post_message("Maciek & Erik - Michelle test", ...)`, and a morning item reads what the other
Keeps posted (their posts land in the history as Telegram Feed rows) and saves what fits.

The spoken destination is a goal in its own right, not a side effect. We **want** the assistant to
be able to start speaking in the kitchen: "let me know when the pasta is done" said at the stove
should end, eight minutes later, with her voice from the kitchen speakers, not with a Telegram
bubble. Today a timer like that can only reach the phone. The same goes for "tell me when the car
is charged" while cooking, or a spoken heads-up that it starts raining in ten minutes while
someone is in the room.

## What already exists, and what history teaches

- **Group plumbing** is in place since 1ce56e1 / f477589: a group on `telegram_bot.allowed_chat_ids`
  is served, the bot answers replies to its own bubbles, a name said in the group reaches her
  through the Telegram Feed (`TelegramFeedIntegration._wake_assistant`), a non-owner is a guest
  with every command disabled, a bot sender is told "reply NO unless you have something to add",
  and `BOT_WAKE_CAP` brakes two assistants waking each other. None of that changes.
- **A bot never sees another bot's messages** — a Telegram Bot API rule, in every privacy mode.
  The owner's personal account does, so the receiving side of the board is the Telegram Feed and
  the history, never the bot.
- **Chat sends exist**: `ChatHub._send_text` → `ChatTransport.send_text(conversation, html)`;
  `TelegramTransport.owner_conversations()` / `notification_conversations()` already split the
  positive (owner) ids from the negative (group) ids on the allow list.
- **The autonomous delivery tail is the mess this plan removes**: `_run_commands_and_extract_message`,
  "only the LAST iteration's text matters for Telegram", a NO regex and "one more extra check for
  flakey LLMs" — all there because text and commands share one reply.
- **Spoken autonomous messages existed and were removed.** In the old repo: fc1ea8d7 (2025-11-01,
  "Server now speaks autonomous action messages", `ServerSpeachPlayback.speak_text`), commented out
  two days later ("they're getting annoyed at home"), re-enabled "only when I'm home", gated to
  daytime (1c0894e2), left commented out in the DialogManager unification (27a3bf4a) and deleted as
  having no caller (b9d7a438, 2026-08-26). The lesson: speaking *every* autonomous message is wrong,
  and blunt gates (daytime, presence) were the wrong fix. With a destination per message the model
  chooses voice only when a spoken line is wanted. The parts the old class used are all still here:
  `VoiceSynthesizer.start(text_iterator) -> audio chunks`, `ServerPlayback.enqueue` /
  `finish_reply` (built per handler by `VoiceInterface.playback_factory`), `VoiceTurnGate`,
  `TimeOfDay`.

## Design

### Destinations (`src/assistant/Destinations.py`)

A registry on the assistant side — the assistant never imports `interfaces.*`, so the interfaces
register what they can deliver to when they start, the way `set_notifier` hands over the chat hub.

```python
@dataclass
class Destination:
    name: str          # what the LLM writes: "owner", "Maciek & Erik - Michelle test", "kitchen"
    kind: str          # "chat" | "group" | "voice" — decides the rendering
    send: Callable[[str], bool]
    aliases: tuple[str, ...] = ()   # the group's chat id as text, a lowercase title
```

- `register(destination)`, `unregister(name)`, `names() -> list[str]`, `describe() -> str` (the
  line the prompt shows: `owner (your chats), "Maciek & Erik - Michelle test" (Telegram group),
  kitchen (speakers)`), `send(name, text) -> str` ('' or the reason it failed).
- Lookup is case-insensitive over names and aliases.
- Three registrations to begin with:
  - **`owner`** — the chat hub: what `notifier().send_message` reaches today
    (`notification_conversations()` of every transport — on a Telegram bot `notify_chat_ids`, the
    proactive route the owner tuned, not every owner chat). Always present while a chat interface runs.
  - **one per Telegram group** on `allowed_chat_ids` (the negative ids) — registered by
    `TelegramTransport` at start, named by the group's title from one `get_chat(chat_id)` call per
    group, alias the id as text. A title change is picked up at the next start; the id always works.
  - **the server speakers** — `VoiceInterface` registers `local_outposts.location` lowercased
    ("kitchen") when local outposts are enabled. See *Voice destination*.
- Rendering follows the kind: a chat or group gets the text as Telegram HTML through
  `RichText` (split at `max_message_length` as `_send_text` does); voice gets the plain text with
  tags stripped.

### The command (`Commands.cmd_post_message`)

`post_message(destination: str, text: str)` — a core command (the bot is an interface, not an
integration), in `MULTI_PARAM_COMMANDS`, documented in `res/prompt.md`'s COMMAND SYSTEM block
beside `remember` (and in `Toolkits.CORE_COMMANDS`; mirror into `testsuite_prompt.md` where it is
linked). Semantics:

- The text is passed through `redact_for_storage` first, like `save_procedure`: a group and a
  speaker are not the owner's eyes, and `VaultLLM` may have restored a real value in the reply.
- `Destinations.send` delivers; the command result is `posted to <name>` or `post_message failed:
  unknown destination 'x' — available: ...` so the model can correct itself in the same turn.
- Not in `SENSITIVE_COMMANDS`: an approval card would defeat the nightly run. The ordinary command
  notification in the owner's chat is the whole audit trail (`send_command_notification` already
  shows every command the autonomous check runs, arguments included) — a post to a group or a room
  leaves no extra copy of the text in the owner's chat.
- No caps per destination or hour: the command notification makes a chatty check visible, and
  `BOT_WAKE_CAP` already brakes the one loop two assistants can get into.
- Logging: `info` with destination, kind and length; `trace` of the redaction count when non-zero.
- `send_telegram` (the owner's personal account, to any contact) stays a separate thing: that is
  the owner speaking, this is the assistant.

### The autonomous check emits commands only

`AUTONOMOUS_ACTION_PROMPT`, *Response Format*, becomes:

> After `</thinking>` output only commands. Anything you want said goes through
> `post_message(destination, text)` — to `owner` for the user's chats, to a group by its name, to a
> room by its name when someone is there and the matter is worth a spoken line. Destinations now:
> {destinations}. Nothing to say and nothing to do: output nothing.

The delivery tail (`_activity_check_inner` after the command loop) shrinks to: run the commands
and save the result to history as today. Text left over after the commands is not delivered
anywhere and gets no handling of its own — we assume the model follows the format, as it does for
every other command-only instruction (the curator's command block, the watch-list polls). A
`trace` line noting that stray text was dropped is enough to see it in the log if the assumption
is ever wrong. The NO convention, the NO regexes and `_run_commands_and_extract_message`'s text
return go in the same change.

The history row stays: `_save_autonomous_result_to_history` keeps writing the whole reply, so the
TUI Home tab and the next check see what was posted where (a `post_message` row is a command row).

### Voice destination (`VoiceInterface.announce`)

This is what makes "let me know when the pasta is done" work end to end: the request is a voice
turn in the kitchen, the assistant schedules a wake-up (`schedule_wakeup`), the autonomous check
fires and answers with `post_message("kitchen", "The pasta is done.")`. The prompt tells the model
that a request made by voice, about something happening in the room, is answered by voice.

`announce(text) -> bool`:

1. Ask the turn gate: a turn is live or a reply is playing → wait up to ~20 s, else log and return
   False (the command result says so, the model may post to `owner` instead).
2. `playback = self.playback_factory(is_killed=lambda: self._stopping)`,
   `voice = create_voice(...)` as the handlers do (TimeOfDay picks the night voice as for a reply).
3. `for chunk in voice.start(iter([text])): playback.enqueue(chunk)`; `playback.finish_reply(text)`.
4. Mute/unmute the outposts around it as a reply does (`ensure_outposts_muted` /
   `restore_outposts_if_muted`), so the kitchen mic does not hear her and wake her.

No daytime or presence gate in code, and no cap: that is the model's call per message, and the
prompt says so.

### The board, once the above exists

The curator posts, a watch item receives. The split keeps the curator's addition to one thing — it
is complex enough — and puts the reading where searching is cheap.

**Posting — the curator** (`ai.memory_curator.tips_destination`, the group's title). The nightly
run is the one component that reads a whole day without a window: the autonomous check sees the
last 15–35 KB of history, on prod an hour or two of a busy evening, while a day is 200–500 KB.
With the key set, the curator's prompt gains one paragraph — go over the period for what another
household's Keep would find useful and post each as `post_message("<group>", """Tip from
<owner>'s Keep: ...""")`, never the household's own people, places, accounts or vault — and
`post_message` joins the commands its block may run, routed to `Destinations.send` with the text
vault-redacted and the destination forced to the configured group. Unset, the paragraph is absent
and a `post_message` line is skipped like any other command that is not the curator's.

**Receiving — a daily watch item**, created by talking to her. Both bots are called Michelle and
the feed journals our own bot's group posts too, so the "Tip from <owner>'s Keep" prefix is what
tells the households apart:

```
save_watch("board-digest", """Every morning: search_history("Michelle test", sort="newest") for the
tips other Keeps posted in the group since yesterday (they start with "Tip from <name>'s Keep",
skip our own). Save what this household could use: remember(...) for a one-liner,
save_procedure(...) for a how-to. Tell the owner only what you kept.""", "2026-10-14 06:30", "1d")
```

The feed rows read `From Michelle to Maciek & Erik - Michelle test: ...`, so the group's title is
in the content the history search indexes.

Prerequisite on prod: `telegram_feed.blacklist` holds `michelle`, which matches the group's title
and Erik's bot's name, so today the board never reaches the history. The entry exists so the bot's
own private chat is not journaled twice; it should name that dialog instead (the bot's username).

## Phases

**Phase 0 — prod, no code.** Fix the feed blacklist entry. Confirm in `log/floxy.out` that the
group's messages arrive as `telegramfeedloop` rows.

**Phase 1 — destinations and `post_message`, chat kinds, commands-only check.**
- `src/assistant/Destinations.py` (new), `src/interfaces/chat/ChatHub.py` registers `owner`,
  `src/interfaces/telegram/TelegramTransport.py` registers the groups (one `get_chat` per group at
  start), `src/assistant/commands/Commands.py` gets `cmd_post_message` + `MULTI_PARAM_COMMANDS`,
  `src/assistant/commands/Toolkits.py` `CORE_COMMANDS`, `res/prompt.md` (+ `testsuite_prompt.md`
  mirror when linked), `AutonomousActionManager` prompt and tail, `config.example.yaml` nothing new.
- Tests: `tests/unit/assistant/test_destinations.py`; the command's lookup, redaction and failure
  text; the hub's and transport's registrations; the manager's tail — a reply of commands only
  sends nothing to the owner, stray text is dropped, `post_message("owner", ...)` reaches the
  owner's chats; the prompt names the destinations. Replace the existing autonomous manager tests
  that assert the Telegram text path.
- Docs: `docs/atlas/descriptions.py` needs nothing for a module with a one-sentence docstring;
  CLAUDE.md's Request Flow gets one line on destinations.

**Phase 2 — the voice destination.**
- [x] `interfaces/voice/Announcer.py` (the method grew past the interface, so it is a module of
  its own): the announcement, its registration under `local_outposts.location`, the outpost mute,
  the wait for a quiet house (`Assistant.is_busy()` or a reply holding the outpost mute).
- [x] The prompt rule that a voice request about the room is answered by voice (phase 1).
- [x] Tests with stand-in synthesizer and playback (`tests/unit/interfaces/voice/test_announcer.py`).
- [ ] A real listen on the development machine before release — "let me know when the pasta is
  done" from the kitchen mic is the acceptance test.

## Out of scope, maybe later

- A "worked for us" reply from the receiving Keep back to the board; a per-peer inbox file; a
  structured procedure exchange — all considered and dropped in favour of plain language in the
  group, the curator's judgement and the history as the inbox.
- `scheduled_prompts` in config (a cron for prompts with a destination) — the watch list already
  is that scheduler; revisit only if items prove too easy to erode.
