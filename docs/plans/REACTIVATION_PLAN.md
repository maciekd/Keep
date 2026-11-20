# Plan: the assistant writes to an owner who has gone quiet

Written 2026-11-19. Status: built 2026-11-20, off by default (`reactivation.enabled`) — the developer runs it
on his own install for a week or two before the default is decided.

## The problem

A read of the beta's telemetry on 2026-11-19 looked for messages sent to an owner who had stopped
writing. Of some 1,570 proactive messages, 40 came a day or more after the owner's last message
and 8 after three days or more; 16 of the 40 were the "Server is up again" note of a restart. One
message in the whole beta was a deliberate "come back": the periodic check's own idea, a day and
a half into a stalled setup. Everything else the check said to a quiet owner followed an open
thread of the conversation, and once those ran out it said nothing — on one install for 13 days
while the server kept running.

Nothing measures the silence, so nothing reacts to it. The two installs that ran on after their
owner left had no phone chat either; the first-run brief now sets Telegram up first, so a later
install can be reached.

## Design

Code decides *when*, the model decides *what*. The periodic check (`AutonomousActionManager`,
every 55 minutes) already holds the whole history and speaks through `post_message`; when a step
of the schedule below has come, the check's instruction gets one more note, and the message it
then posts is the reactivation message.

### 1. What counts as silence

The clock is the owner's last turn: a request that is neither a guest's nor a turn the server
started (`SYSTEM_USER` — the brief, the resume turn, "pressed Start"). `Assistant.stream` already
tells these apart where it sets `spoken_to`; the same line calls `Reactivation.owner_spoke(source)`.
Any way in counts — a chat, the console, a voice — so a household that talks to its speakers
every day is never written to.

What the assistant sends on its own (a watch item's reminder, a restart note, the brief a restart
re-opens) neither counts as the owner's activity nor as a step of the schedule.

The time is kept in `cache/reactivation.json` so a restart does not start the clock over, and it
is kept whether the key is on or not, so switching the key on finds a clock that is true. An
install without the file (the first start with this code, a `//wipe`, one nobody has ever spoken
to) has no clock until the owner's next turn and is not written to: the history table holds the
feeds' rows beside the dialog's, so the last turn cannot be read back out of it, and a message
must not claim a silence nobody measured. An owner who went quiet before this release is
therefore out of its reach.

A `//wipe` deletes the file, and no backup holds it: the clock is a measurement of the running
install, and one a restore brought back would count a silence the restore itself had ended —
the assistant would write within the hour to an owner who had just been there. A restore leaves
the clock the install has, which is the turn that asked for it.

### 2. The schedule

Counted from the owner's last turn and due an hour ahead of it (`LEAD`), so each message is there
a little before the time of day they were last active. The minute does not matter, only that it
is early on average:

| Step | Due after | Note |
|---|---|---|
| 1 – 6 | 1, 2, … 6 days, less an hour | the daily note |
| 7 | 7 days, less an hour | the back-off note: the last daily one — the daily note when none of the six went out |
| 8, 9, … | 21, 35, … days (every 14), less an hour | the fortnightly note, for as long as the server runs |

- A step is noticed by the first periodic check after its time, so up to 55 minutes late: the
  message lands between an hour and five minutes before the owner's time of day, half an hour
  before it on average. (A lead of 30 minutes would land it on the owner's time on average, as
  often after as before.) In Night mode the periodic check does not run, and the step waits for
  the morning.
- One message per check and none within 12 hours of the last: after a server that was down for
  days, the steps that passed meanwhile are one message, and its step is the newest that is due —
  the back-off comes on day 7 by the calendar, however few of the six before it went out.
- The back-off calls the daily messages off, so it needs one that went out: when none did (the
  key switched on a week into a silence, a server that was down, six steps the model declined)
  step 7 carries the daily note and is reported as `daily`. The next message is still 14 days on.
- A turn of the owner's ends it: the schedule starts again from that turn.
- Only the periodic check carries the note — not a wake-up, a watch item's check or a caffeine
  check — and a check that gave way to the user is retried without it (the owner just spoke).

### 3. Where it goes

The model decides. The check's instruction already lists the destinations `post_message` reaches
(`Destinations.describe()`: the owner's chats, a person's iMessage chat with the assistant, a
room's speakers, a group), and the note asks for the one where the message is most likely to be
seen and welcome. No code picks a channel and no code holds a step back for want of one.

The note does not say what `owner` amounts to at the moment — Telegram on a phone, or a console
pane nobody has open. A first draft did, in a line built from `Notifier.owner_channels()` (on the
`ChatHub`, each transport with a notification conversation by its name, and whether somebody can
see it now); it went with the rest of the long note (see 4). The lookup stays for the telemetry.

The delivery stays within `post_message`. A mail or an iMessage sent *as the owner* to the owner
is not a way to reach them: those commands act on the owner's own accounts and wait for an
approval that would itself land in the chat nobody reads.

### 4. The note

Appended to the check's instruction as a turn of its own, like `DUE_ITEMS_NOTE`, and written by
the format-neutral rules (a command is its bare backticked name). It is the fact and the ask,
and the model works out the rest. A longer draft — what to look for, how the message must not
read, how to choose the destination, what `owner` reaches — was cut on 2026-11-20:

> ## QUIET OWNER
>
> {name} last wrote {duration} ago.
>
> Send them one message with `post_message` that gives them a reason to come back. One
> destination, one message. If the history or the calendar says why they are away and a message
> would be unwelcome now, or nothing listed would reach them, answer NO.

Where the schedule stands is the one thing the model cannot work out, so the back-off (step 7,
when a daily message went out before it) adds a sentence after the first: this is the last
daily message, after it one every 14 days, and the message says so. The daily and the
fortnightly messages carry no position.

The row that opens the check's rows in the history (`_opening_note`) says what the check was:
"reactivation message 3 of 7, the owner has been quiet for 3 d; the config key
`reactivation.enabled` turns these off". A later chat turn then knows what its own message was,
and what to set when the owner says "stop writing to me".

### 5. What a step leaves behind

After the check, the step is recorded with what came of it — `sent` when a `post_message` ran in
the check, `declined` when the model answered NO — and is spent either way: a declined step is
not asked again 55 minutes later. Any `post_message` of the check counts: a check that carries
the note and posts a watch item's reminder has written to the owner, which is what the step is
for. A check that gave way to the user or failed before it posted leaves the step due, and so
does one the model gave no answer to — a safety refusal comes back as an empty reply, which is
not a NO, and counted as `declined` it would cost the day's message and blur the one number
that says how often the model chooses silence. One that was cut short after it posted has spent
it. A step found due before the owner spoke and recorded after is reported and not kept — the
schedule it belonged to is gone.

### 6. Telemetry

Two events, neither carrying text (the message itself is the `assistant_turn` with
`proactive: true` of the same moment, under the seven-day transcript rule as always):

| Event | When | Properties |
|---|---|---|
| `reactivation_nudge` | a step was spent | `step`, `kind` (`daily` / `backoff` / `fortnightly`), `outcome` (`sent` / `declined`), `via` (the kinds of the destinations posted to — `chat`, `voice`, `group` — never a name), `owner_channels` (the names of the transports a push to `owner` reaches), `inactive_hours` |
| `reactivation_return` | the owner's first turn after a silence of a day or more | `inactive_hours`, `nudges_sent`, `last_step`, `hours_since_nudge` (absent without a nudge), `source` |

`reactivation_return` goes out with `nudges_sent: 0` too — the key off, every step declined, a
return before the first step — which is the base rate the nudged returns are read against.
`via` and `owner_channels` are what shows whether the model's choice of channel matters: a
return after a message to a phone against one after a line in the console.

### 7. Config

`reactivation.enabled`, false by default, read at every check; documented in
`config.example.yaml`. The assistant may set it (it is the owner's own preference).

## The pieces

- `src/assistant/Reactivation.py` — new: the schedule, the state file, `owner_spoke`, `due`,
  `record`, the note. Imports nothing of `interfaces`; reads the clock through an argument
  so the tests need no sleeps.
- `Assistant.stream` — the `owner_spoke` call beside `spoken_to`.
- `AutonomousActionManager` — `_run_periodic_check` asks `Reactivation.due()` and hands the step to
  `activity_check`; the note, the opening row, the record after the check.
- `Notifier` / `ChatHub` — `owner_channels`, for the telemetry (the `NullNotifier` answers with none).
- `Telemetry` — the two events; `.claude/skills/ftue-report/references/event-model.md` lists them.
- `config.example.yaml` — the key.
- `DebugCommands._wipe` — deletes the clock with the state; it is not in `Backups.STATE`.
- Tests in `tests/unit/assistant/test_reactivation.py` (the schedule over a fake clock: the hour
  of each step, the 12-hour gap, the collapse after a gap, the reset on a turn), plus
  the check carrying the note and recording the outcome and the kinds
  posted to, the hub's `owner_channels` with an unattended console, and the two events'
  properties as a contract with PostHog.

## Open points


1. **The fortnightly messages never end.** As asked. An owner who has left for good and whose Mac
   keeps running gets one every two weeks until they block the bot or stop the server. A cap
   (say, six of them) would be one constant.
2. **One threshold for everybody.** A set-up owner who goes on holiday for a week gets the seven
   daily messages like a tester who stalled in setup. The NO in the daily note is the only thing
   that softens it. The alternative is a first step that waits longer once setup is over.
3. **Step 7 is both the seventh message and the back-off.** Duolingo sends its "these reminders
   don't seem to be working" as a message of its own after the week.
4. **The default.** Off for now. Turning it on for every install is the default of `due()` and
   the value in `config.example.yaml`; an install that has been silent since before this release
   has no clock and stays out of reach either way.
