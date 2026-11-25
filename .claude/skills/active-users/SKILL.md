---
name: active-users
description: Build the "Who talked to the assistant" HTML page from Keep's PostHog telemetry - a heat-shaded table of how many messages each active install sent to the AI per day and how many Keep sent on its own (days as rows, installs as columns, totals on both edges). Use whenever the user asks who has been active, how much the testers use Keep, for active users or interactions per user per day, or for a usage table, grid or heatmap over the last days - even if they don't say "PostHog".
disable-model-invocation: false
allowed-tools: Bash(python3 *), Bash(security find-generic-password *), Bash(open *), Bash(ls *), Read, Write
argument-hint: "[window, default 7d] [output path]"
---

Write a self-contained HTML page showing how many messages each active install sent to the
assistant per day, and how many the assistant sent on its own. Argument: $ARGUMENTS - a window
(default the last 7 calendar days, today included) and optionally where to put the page (default
`~/Desktop/keep-active-users.html`).

## What the page answers

Two numbers per install and day, Keep's in front: the messages Keep sent on its own and the
`user_turn` events of that `distinct_id`, on calendar days in Stockholm time. An install is
active when it has at least one `user_turn` in the window; columns are ordered by their total,
largest first, and the shade follows them too. Days with no message show as empty cells, and
today is a partial day - say until when in the lede.

Keep's own messages are the `assistant_turn` events with `proactive = true`: what a check, a
wake-up or a report posted, never a reply. The notices of a restart or an update are reported
the same way and are no messages: leave out every one within 20 seconds, before or after, of a
`heartbeat` with `startup = true` of the same install (measured 2026-11-25 over a week: half of
the proactive turns, "Server is up again", "PR trial: Switched to" and the like, each within 17
seconds of a start, and the nearest message that was none of them 33 seconds from one).

A cell where the install was sent a reactivation push that day is drawn in the push colour: the
`reactivation_nudge` events with `outcome = 'sent'`, on the same Stockholm calendar days (a
`declined` or `covered` step sent nothing).

Hovering a cell shows the last message the person wrote that day, the last one Keep sent on its
own (left out when it is the push) and the push's text, where telemetry holds them: an
install's turn text is recorded during its first week only. A turn the server started itself is
nobody's message.

## Facts that hold

- Query PostHog with `.claude/skills/ftue-report/scripts/phq.py` (HogQL on stdin; the key is
  read from the keychain and never printed). It needs `POSTHOG_PROJECT_ID`; when that is unset,
  the project id is in the session's memory notes or the user's to give.
  `.claude/skills/ftue-report/references/event-model.md` describes the events.
- A column is an install, not a person: a household is one column, and one person may run
  several installs.
- `user_turn` also counts the turns the server starts itself during setup, so a new install's
  first day reads high.
- Timestamps are stored in UTC.

## Naming the columns

A column is headed by its owner's first name, with the first eight characters of the
`distinct_id` under it and a short tag where it helps (`prod`, `new, Linux`, a test box). Use
the names the memory notes already hold. For an id nobody has named, look at all of its
history, not only the window: the reply to the setup's "owner_name is not set" turn is the
name (turn text is recorded during an install's first week and stays queryable), and an
install that records no text may carry a person property set by a `$set` event. An id that
stays unnamed is headed by the id alone. Save a newly found name to memory.

These names are private: they go into the page and the memory notes, never into this skill, the
repository or a commit.

## Writing the page

Copy `assets/template.html` and replace its `const data = {…}` object - `lede`, `notes`,
`users`, `days` (a day holds, per install, the person's messages and then Keep's own; its
optional fifth array the step of the push sent that day, 0 for none; its optional sixth the hover
texts, as the sample shows) - and nothing else; the sample values there are fictional and show the shape.
The stat tiles, shading and totals are computed from it. Then open the page and tell the user
the path, the totals, who the installs are and anything that stands out (a new install, one
gone quiet, one Keep wrote to and got no answer from, how much is the user's own machines).
