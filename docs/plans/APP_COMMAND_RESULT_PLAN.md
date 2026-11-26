# The result behind a command row, in the app

## Why

The app's chat shows each command as a row of a card: what ran, how long it took, how much came
back ("4s · 1.2k"). What came back is not there. The owner who wonders why the assistant
answered as it did — what the search found, what the car said, why a command failed — has to
open `log/command_replies/` on the server.

A finished command row whose command handed something back is tapped, and a sheet shows it: the
text the model read, and the pictures. This is the first item of phase 4 in `APP_CHAT_PLAN.md`
("the result behind a command row").

## What a result is, and where it is today

A command reaches the model's next round one of two ways:

| Path | Result | Where the echo is completed |
|---|---|---|
| handed to a lane (`Commands.background_command`, 94 call sites: every read, search, fetch) | the function's return value — a text, or a block list of texts and images | `Commands._run_one` |
| run inline on the command worker | nothing on success; an error text on failure | `CommandProcessor._run_echoed` |

Both complete the echo through `notifier().complete_command_notification(echo, detail)`, and
`detail` is words only ("done after 4s, received 345 chars"). The result itself goes on to
`CommandProcessor.set_command_reply`, merged with the round's other results — from there on
nobody knows which command it belonged to. So the result has to travel with the echo, at the two
places above, where it is still one command's.

## The server

A result can be 200 KB, so it is never sent with the chat: the server keeps it in a file when the
command ends, the item carries only the route to it, and an app fetches it when the row is tapped.

**Carried with the echo.** One optional argument, end to end:

- `Notifier.complete_command_notification(handle, detail, result=None)` — `result` is what the
  command handed back for the model: a text, a block list, or None.
- `Commands._run_one` passes its `result` (after a raise: the "Command failed: …" text, which is
  what the model is told as well). `CommandProcessor._run_echoed` passes the error text of an
  inline command that failed, and None for one that succeeded.
- `ChatHub.complete_command_notification` passes it on to
  `ChatTransport.finish_command_line(conversation, ref, text, detail, result=None)`. The hub
  keeps no copy. The console and Telegram take the argument and ignore it.

**Kept by the app transport**, beside the log, as the photos are. `AppTransport.finish_command_line`
turns the result into two fields of the `command` item:

| Field | Holds |
|---|---|
| `result` | the route to the result's text, `/app/media/<id>.txt` — `""` when it has no text |
| `images` | the result's pictures, each `url`, `width`, `height` — the field a `user` item has |

- The text is the string, or a block list's text blocks joined by a blank line; the image
  blocks are decoded and kept with `_keep_picture`.
- It goes through `redact_for_storage` first: a vault value is in the file as its placeholder,
  as in everything else the assistant writes to disk.
- It is cut at `RESULT_CHARS` (100 000), the cut ending in a line that says how long the whole
  was. `detail` keeps the true size.
- A result that is empty leaves both fields empty: the row is not tappable, which is the truth
  for an inline command that worked.
- Keeping it must never cost the row its ending: a failure to write is logged and the item is
  finished without a result.

Nothing new is needed around it. The text is served by the existing `/app/media/<id>` route (a
week of `Cache-Control`, so a phone asks once); `Transcript._forget_media` forgets `images`
already and gets `result` added to its addresses, so a result goes when its item is trimmed
(500 items, 7 days); `_settle` copies an item's fields whole, so a restart keeps them. The
result is not in the item, so the event stream, the replay and `app_chat.json` stay small.

Worst case on disk: 500 results of 100 KB. Typical is a few KB.

**Who can read it**: whoever reaches the address, as with the rest of the chat — a result is
shown where its command was echoed, and that is an owner's conversation.

**Compatibility**: an app older than this ignores `result`; a server older than this sends
none, and the app's rows stay as they are. Neither needs the other updated first.

## The app (`~/Documents/Keep-Apple`)

- **`ChatItem`**: `var result: String?`, and `hasResult` — the command is not running and has a
  `result` route or `images`.
- **`CommandRow`**: with a result the row is a button over its whole width, lit while pressed, a
  dim chevron after the outcome. A running row and a row without a result do nothing; in a card
  where some row opens, every row leaves the chevron its place, so the outcomes stand in one
  column. A folded card opens first, as today.
- **`CommandResult.swift`** (new) — the sheet, presented from `CommandCard` (`.sheet(item:)`),
  at half height with a drag to full: the preview and the view in one.
  - Head: the glyph, the command's text, the outcome in words ("done after 4s").
  - The pictures (`item.pictures`), each opening in `MediaViewer`: a row of tiles over the
    text, or — a result of pictures alone — at the size a message's have.
  - The text: fetched when the sheet opens, plain, monospaced, selectable, in a `UITextView` —
    a `Text` lays out all 100 000 characters at once. No markdown, no HTML, no tappable links:
    it is what the model read, and much of it (a mail, a web page) was written by an outsider.
  - Copy and Share in the bar.
  - While it loads: the working line's shimmer. Unreachable: the reason and Retry. A 404: "This
    result is no longer kept."
- **`ChatConnection.result(at:)`** — a GET of the item's route (`ChatItem.resultURL`), UTF-8
  text back, a 404 as `Failure.gone`. The URL cache holds it; the app keeps no store of its own.
- **`ChatSamples`**: a command with a result and one without in the recorded stream, and a
  preview of the sheet over a text given inline.
- `CLAUDE.md` there: the new file in the sources list.

`ChatItemViews.swift` has an uncommitted change of another session (the folding thought); the
row's change is made beside it.

## Tests (server)

`tests/unit/interfaces/app/` and beside the hub's and the commands' tests:

- a lane command's text reaches `finish_command_line`, and the item's `result` route serves it
  through the media route; a failed one serves the reason
- a block list: the texts joined, the pictures in `images` with their sizes
- an inline command: no result when it worked, the error when it did not
- a vault value in a result is its placeholder in the file
- a result over the cap is cut and says its whole size; an empty one leaves the item without a route
- a trimmed item's result file and pictures are gone; a restart's `_settle` keeps the fields
- a write that fails still finishes the row
- `result` and `images` on a `command` item are a contract with the apps: pinned, and named so

## Order

1. Server: the argument through notifier, hub and transports; the transport's keeping; the
   tests; `result` and `images` in the item table of `APP_CHAT_PLAN.md`, and its phase-4 line
   pointing here.
2. App: the item field, the row, the sheet, the fetch.

Each ships alone.

## Not in this plan

- **The call itself** (`car_status(KEP 47L)`) in the sheet. The row shows the description; the
  parameters can hold a secret before the vault knows it (`vault_add`), so the raw call needs a
  rule of its own.
- **The console.** The hub now hands its transport the result too; a view for it is its own change.
- **Search inside a result**, and anything that renders it (tables, JSON folding).
