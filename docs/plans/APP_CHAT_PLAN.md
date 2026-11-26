# The phone app as a chat of its own

Status: phases 1 and 2 built 2026-11-24, on the server and in the iOS app, and run in the
simulator against the real transport with a scripted assistant — not yet on a phone or against
a real Keep. Phase 3 (pictures in) built 2026-11-25: the app's own model and connection were run
against the real routes, a message with pictures sent from the simulator — the photo picker, the
camera, paste and drop have not been used by hand yet.

Today the iOS app's chat posts to `/stream` and reads back plain text: no commands, no working
line, no thinking, nothing the server says unasked, no approvals. The console and Telegram get
all of that because they are `ChatTransport`s on the `ChatHub`. This plan makes the app the
third one — `interfaces/app/` — and gives the app a screen that draws what a transport is told.

The app's v1 (its `CHAT_PLAN.md`, in the app's repository) carries over: the screen, the
composer, the palette, `RichText.swift`. Only `ChatClient` and the flat `Turn` model are replaced.

## The one idea: the server owns the transcript

A transport is told things in small imperative calls from many threads (`begin_draft`,
`update`, `send_command_line`, `finish_command_line`, `send_approval_prompt` …). The console
turns them into `Entry` objects in a pane. `AppTransport` does the same, into a **transcript**:
one ordered log of items, shared by every phone, each with an id, where every change is "this item is now
this". The app renders the list and knows nothing else.

What that buys:

- **Reconnect is replay.** The app says the last sequence number it saw and gets what it missed —
  after a tunnel, a lock screen or a relaunch, also half-way through a reply.
- **What the server said unasked is there at the next launch**, with no push service.
- **No matching heuristics.** The app's own bubble comes back carrying the id the app gave it.
- **Android gets the same chat** by drawing the same items.

## The wire

Under `/app`, registered on the HTTP interface's Quart app. No call names the phone: there is
one conversation, and the cursor is the app's own.

**`GET /app/events?after=<seq>&log=<log>`** — one long response, a JSON object per line:

| Line | Meaning |
|---|---|
| `{"log": "<id>"}` | first: the log the sequence numbers are counted in; the app sends it back with its cursor |
| `{"seq": n, "item": {…}}` | the item with this id is now this (new or changed) |
| `{"seq": n, "remove": "<id>"}` | the item is gone (a discarded draft) |
| `{"working": {…}}` / `{"working": null}` | the working line; not logged, not replayed |
| `{"reset": true}` | the cursor is of another log or older than this one reaches: drop the local copy, a full replay follows |
| `{"live": true}` | the catching up is over; what follows happens while the app watches |
| `{}` | every 15 s, so a dead connection is noticed |

An item: `id`, `kind`, `at`, `order` (its place in the list: the sequence number it first
appeared under, which a later change does not move), and by kind —

| Kind | Fields | Comes from |
|---|---|---|
| `user` | `text`, `images` (each `url`, `width`, `height`; the bytes under `/app/media/`) | `/app/send` (the id is the app's) |
| `assistant` | `html`, `final` | a `Draft`: `update` / `finalize` / `discard` |
| `thinking` | `text`, `done`, `seconds` | `Progress.thinking_text` |
| `command` | `text`, `state` (`running` `done` `failed`), `detail`; once it has ended `result` (the route to the text it handed back, under `/app/media/`; `""` for none) and `images` (its pictures, as a `user` item's) | `send_command_line` / `finish_command_line` |
| `notice` | `html` | `send_text` — `post_message`, the check's findings, apologies |
| `media` | `media` (`photo` `video`), `url`, `poster`, `caption`, `width`, `height`, `album` | `send_photo` / `_album` / `send_video` |
| `approval` | `token`, `title`, `details`, `outcome` | `send_approval_prompt` / `close_…` |
| `secret` | `token`, `title`, `details`, `outcome` | `send_secret_prompt` / `close_…` |

`working` is `Progress` as it is: `phase` (`waiting` `prefill` `queued` `thinking` `commands`),
`elapsed`, `output_tokens`, `rate`, `prefill`, `commands`. The app words it; the server sends facts.
One phase is the transport's own, `replying`: a reply is being written or a round just ended.
There is a working line for as long as a turn is on, so it is also what the stop button follows.

A streaming `assistant` item is sent whole on every change, at most every 50 ms (the newest
state wins). Whole, not appended: `RichText.repair_markdown` rewrites text that was already sent.

**The app's side** — plain POSTs, so each has a status the app can show:

| Route | Body |
|---|---|
| `POST /app/send` | `id`, `text`, `user` as JSON; with pictures a multipart form of the same fields and an `image` part per picture (10 at most) |
| `POST /app/stop` | — `ChatSession.cancel()` |
| `POST /app/approval` | `token`, `decision` |
| `POST /app/secret` | `token`, `value` (null cancels) — goes to `hub.claim_secret`, never logged, never an item |
| `GET /app/media/<id>` | the bytes of a `media` item, with byte ranges (a phone's video player asks for them); the text of a command's result |

An answer to a question that is no longer open is a 410. A server without the routes answers
404 on `/app/events`: the app then says the server needs an update.

A message's pictures are re-encoded on the way in (`picture_as_jpeg`: any format, 2048 px at
most) and go to the session as one photo message with the text as its caption. A send under an
id the transcript already holds is answered 200 and starts no second turn — an app that did
not hear the first answer sends again. Too many pictures or one that cannot be read is a 400
with the reason; a server older than pictures answers a form with 400 `empty message`, which the
app reads as "needs an update".

## The server

`src/interfaces/app/`, an `Interface` like the others (`interfaces.app.enabled`, on by default:
it adds routes, and nothing reaches them without `interfaces.lan_access`).

- **`Transcript.py`** — the item log: upsert, remove, sequence numbers, replay
  from a cursor, subscribers, a cap (500 items or 7 days), persistence in `cache/app_chat.json`, the media beside it.
  No Quart, no hub: this is where most of the tests are.
- **`AppTransport.py`** — the `ChatTransport` and its `AppDraft`. One conversation, `"app"`, the
  owner's — as the console has `"tui"` (a phone is the owner's, as every HTTP client is
  today); every phone shows the same list, and a message typed on one appears on the others.
  `attended()` is true while a phone holds an event stream, so a
  turn the server starts prefers a phone that is being looked at over a console nobody sees.
  `max_message_length = 0`, `keeps_partial_reply = True`.
- **`routes.py`** — the blueprint. Transport calls come on worker threads and reach the
  streams through `loop.call_soon_threadsafe`. A restart closes what the last run left open
  (a reply still streaming, a command still running, a question nobody can answer any more).
- **`AppInterface.py`** — registers the transport on the hub and the blueprint on the app.

Outside the package:

- **`ChannelProfile`**: the `App` row takes the console's values (smart tier, command echoes
  as lines of their own, tokens streamed) and is the transport's. Nothing posts `source=App`
  to `/stream` any more; the app's `ChatClient` goes.
- **`config.example.yaml`**: `app: { enabled: true }` under `interfaces`.
- **Atlas**: a `GROUPS` blurb for `interfaces/app`.

Nothing in `assistant/` changes for phases 1 and 2.

## The screen

Dark, the console's palette, the v1 layout. A turn reads top to bottom the way it happened:

```
                        ╭──────────────────────╮
                        │ is the car charged?  │
                        ╰──────────────────────╯
  Thought for 6s  ›                                dim; tap opens the thinking

  ╭──────────────────────────────────────────────╮
  │ ✓  Loading the cars toolkit            0.2s  │
  │ ✓  Checking the car                4s · 1.2k │  one card per run of commands
  │ ◌  Reading the charging history…             │
  ╰──────────────────────────────────────────────╯

  Running command…  12s                            the working line, shimmering

  78 %, and it will reach 90 by 06:40.             the reply, full width, no bubble
```

- **The working line** stands where the next thing will appear. A shimmer sweeps the label;
  the facts after it are dim and count without jumping (`contentTransition(.numericText)`).
  - `waiting` / `thinking` — "Thinking…", then `↓ 1.2k tokens · 45 t/s · 12s`
  - `prefill` — "Reading…" over a thin cyan capsule bar, `42 % · 1.5k t/s`
  - `queued` — "Waiting in queue…"
  - `commands` — "Running command…" / "Running commands…", the seconds
- **Thinking** is a block of its own above the reply. Live: the last three lines, dim, fading out
  at the top. Done: it folds to "Thought for 6s ›"; a tap opens the whole text.
- **Commands** in a row share one card on `TITLE_BG` with `GRID` hairlines between the rows. A
  row is a glyph, the echo text, the detail on the right: a spinning ring while it runs, a teal
  check when done, a red cross when failed — the glyph morphs (`symbolEffect(.replace)`).
  Once the reply is over a card of more than three rows folds to "5 steps ›".
- **A notice** — something Keep said unasked — is reply text under a dim "Keep · 14:02". What
  arrived while the app was away stands under a hairline marked "New".
- **Photos and videos** are inline with rounded corners and the caption under them, an album as
  a grid; a tap opens them full screen.
- **An approval** is a glass card: title, details, Deny and Approve (cyan). Answered — here or
  in another chat — it folds to one line, "Approved · 14:02".
- **A secret** is a glass card with a masked field and an eye. The value never enters the list
  or the app's log.
- **The right button** stays voice / send / stop; stop now cancels on the server.
- **Scrolling**: pinned to the bottom while the user is there; scrolled up, a round glass "↓"
  appears and nothing moves under the finger.
- **The connection**: a dim "Reconnecting…" under the top bar while the stream is down. A send
  that fails keeps its bubble with the red line and Retry, as today.
- **Haptics**: a soft tick when a reply's first word arrives, a success tap on an approval.

In the app: `ChatConnection` (the stream, the cursor, the POSTs, reconnect with backoff),
`Transcript` (the items, kept on disk so the list is there at launch before the server
answers), one view per item kind, and `ChatScreen` grouping consecutive items into a turn's
cards. `ChatSamples` is a recorded stretch of the event stream the preview is fed from, so the
look is worked on without a server.

## Phases

1. **A typed turn, as the console shows it.** `Transcript`, `AppTransport`, the routes, the
   `App` profile; items `user`, `assistant`, `thinking`, `command`, `notice`; the working
   line; stop; replay. In the app: the connection, the transcript, those views.
2. **What the server starts.** Approvals, secrets, photos and videos out, `attended()`, the
   "New" line.
3. **Pictures in.** The plus button: the photo picker and the camera, paste and drop, staged
   thumbnails over the composer, one request per album as the console sends it.
4. **Later, each its own plan.**
   - The result behind a command row (tap to read it): `APP_COMMAND_RESULT_PLAN.md`.
   - Voice turns in the list: a voice request does not name its phone yet.
   - Push while the app is closed: `APP_PUSH_PLAN.md` (APNs needs a relay, since an install
     cannot hold the app's key).
   - A person per phone, each with a conversation of their own, and a token: today whoever
     reaches the address is the owner.
   - Android.

## Tests

`tests/unit/interfaces/app/`: the transcript (sequence, replay from a cursor, reset past the
cap, the newest state winning in a burst, persistence); the transport driven by a real
`ChatSession` over scripted assistant events — a turn with thinking, two commands and a reply
ends as the expected items; the routes through Quart's test client, the secret route included
(the value in no item and no log line). The item fields are a contract with the apps: pinned,
and named so.

## Decided

- **One conversation for every phone.** No device registry; the dialog history is one anyway.
- **Thinking is shown**, live and then folded.
- **The v1 `/stream` chat is dropped**, not kept beside the new one.
