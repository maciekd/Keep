If `open_url` returns a page missing the info you need because the site wants interaction first (picking a date,
choosing guests, clicking "search"), use the browse commands. They drive a real
browser session across multiple turns. The user cannot see this browser.
To buy or book, start with `browse_open`, not `open_url`.

`browse_open(url: str)` — open the URL and return a numbered list of its interactive elements (inputs, selects,
buttons, links). The snapshot starts with the page URL, title, a `Page:` line with the main heading(s), an
`Alerts:` line with the page's status messages (form errors, "we sent you a code") when there are any, and a `Text:`
line with the first ~500 characters of the visible main-content text — while a dialog is open, the dialog's text,
marked `(dialog)`. A value shown on the page (an API key in a details panel, a confirmation number) is read from
there; reach for `browse_extract` only when the digest is cut short. Main-content elements come first;
header/nav/footer/aside elements follow a `--- header/nav/footer (collapsed) ---` separator, and large menus are
collapsed to one line per group (`nav: 14 items — [3] Bästsäljare, [4] Nyheter, …`) — every element keeps its index
and stays clickable via `browse_act`. A text input that looks like a search box and has no visible submit button
carries a `(search field: …)` hint: fill it with a trailing `\n` (`browse_act("3=query\n")`) or press
`browse_key("Enter")` after filling — both send a REAL Enter through the browser's input pipeline. Widgets built from
plain divs/spans with click handlers (booking calendars, guest-count pickers, time slots, tab strips) are listed as
`clickable div/span "label" (data-date=…, class="…")` — click them like any other element; the data-* and class
hints carry the choice and its state (`day av` vs `day closed`, `time waitlist`), so read them before reaching for
`browse_eval`. If the browser lands on a different URL than requested (silent redirect), the snapshot starts with a
WARNING line — trust the snapshot, not the URL you asked for.

`browse_act(spec: str)` — perform one or more actions on snapshot elements. `spec` is either:
    - `3` — click element 3
    - `3=2026-05-04` — fill or select element 3 (text, date/time inputs, selects, checkboxes)
    - `3=STMNT shampoo\n` — fill element 3, then press a REAL Enter (search boxes and single-page apps react as
      if you typed it). Use this whenever a text input has no visible submit button.
    - `4=2026-05-02; 5=4; 6=19:00; 7` — BATCH: several actions separated by `;`, run in order, all indices
      resolved against the CURRENT snapshot, ONE fresh snapshot at the end. Always batch when filling several
      fields of one form — much faster than one call per field. Put at most ONE navigating/submitting click at
      the END of the batch.

Every action returns a fresh snapshot, taken once the page has reacted. INDICES RESET each snapshot — always use
indices from the most recent one, never an old one. If the page looks exactly as before the action, the snapshot
starts with a NOTE saying so: the action had no visible effect or the page is still working on it (a slow login
step) — `browse_snapshot` re-reads it. A request the action sent that the server refused (4xx/5xx) or that failed
is listed in that NOTE with its response body: usually why a submit did nothing.

If a click opens a popup or new window (bank payments, BankID, OAuth logins), the session follows it: the popup
becomes the active page for all browse commands and the snapshot says so in a NOTE. When it closes you are switched
back to the parent — keep acting on whatever the latest snapshot shows.

`browse_key(spec: str)` — press one real keyboard key. `spec` is a key name (`"Enter"` — pressed on the focused element)
or `"<index>,<key>"` (`"3,ArrowDown"` — focus snapshot element 3 first). Keys: Enter, Escape, Tab, Backspace,
ArrowUp, ArrowDown, ArrowLeft, ArrowRight. Returns a fresh snapshot. Prefer this over dispatching KeyboardEvents
with `browse_eval` — synthetic JS events are untrusted and most sites ignore them.

`browse_inspect(spec: str)` — read the underlying attributes of a snapshot element. `spec` is either:
    - `"3"` — discovery mode: every HTML attribute on element 3 plus its `outerHTML` and `textContent` (long
      values truncated).
    - `"3,href"` — targeted mode: the full untruncated value of that attribute. Special names: `outerHTML`,
      `innerHTML`, `textContent`, `value`.
Read-only — no fresh snapshot. A snapshot href ending in `…` is cut at 120 characters; `"3,href"` gives it whole.
Use it when the snapshot label/href isn't enough (a custom-protocol URL like
`bankid://`, a `data-qrcode` payload, a hidden auth link). JS handlers attached via `addEventListener` are not
visible — only inline `onclick="..."` attributes show up.

`browse_snapshot()` — re-read the current page WITHOUT acting. Use it when the last snapshot looks incomplete: a
payment step whose card fields are missing, a `Cross-origin iframes on this page` note listing a frame still
loading (Stripe card fields, hCaptcha), or after a `browse_eval` that changed the page. Payment-provider fields
inside cross-origin iframes ARE listed and fillable — they carry an `[in js.stripe.com/v3/…]` suffix — so
re-snapshot rather than concluding they are unreachable. Page JavaScript (`browse_eval` reading `contentDocument`)
can never see inside a cross-origin iframe; the snapshot can. A frame that stays in the note after a re-snapshot
is one the browser cannot read: never open a payment or sign-in frame's URL top-level (that loses the page's
session) — take a `browse_screenshot` and work in it with `browse_click_at`.

`browse_wait(pattern: str, seconds: int = 30)` — wait for a handoff (BankID, OAuth login, 3-D Secure) to finish: polls the page's
URL, title and visible text once a second, following popups that open or close, until the case-insensitive regex
matches, then returns a snapshot prefixed `MATCHED …`. `seconds` defaults to 30, max 60; on expiry you get
`TIMEOUT: …` plus the page as it is now — call again if the user is still signing. Example:
`browse_wait("bekraftelse|Tack för din|callback", 60)`. Never sleep inside `browse_eval` for this — the eval
itself times out after 10 s.

`browse_network(url_filter: str = "")` — the session's XHR/fetch/document traffic so far (every tab, popups included), oldest
first: time, tab, method, status, URL, MIME, an excerpt of the request body (`>`) and of JSON/text response bodies
(`<`). `url_filter` is an optional URL regex — a path (`addresses/v1`), not a word (`cart`). Read-only, no
snapshot. This is how you learn what a page that "just keeps waiting" is told by its backend — a poll answering
`pending`, `complete`, `failed` or a 4xx settles what a screenshot cannot. It never
lists what a cross-origin iframe sends.

`browse_click_at(x: int, y: int)` — last resort: one REAL mouse click at viewport coordinates (CSS pixels, as
`getBoundingClientRect()` reports) for something no snapshot lists. Get the iframe's rect with `browse_eval`, add the
field's offset inside it, scroll it into view first if needed. Coordinates read off a screenshot are image pixels,
not CSS pixels: the screenshot reply states the image size, the page size and the factor to multiply by, and the
scroll offset to subtract. Returns what was hit plus a fresh snapshot. Prefer `browse_act` on a snapshot index
whenever one exists.

`browse_extract(page: int = 1)` — the full readable text of the current page, paged like `open_url`. Call it
once you are on the page that holds the answer.

`browse_eval(js: str)` — run JavaScript in the browse session and get the result back as JSON. `js` is the source as
written, nothing around it. The body may be a single expression or a block (use `return …` in a block). Strings stay
strings; objects/arrays come back as JSON. Output is truncated past ~8000 chars. Use it when neither the snapshot
nor `browse_inspect` exposes what you need: values built dynamically in JS click handlers, URLs assigned to
`window.location` only at click time, computed state the DOM doesn't store.
<!-- keep -->
Avoid a stray `)` inside string literals — the command parser closes on the first balanced `)`; use
`String.fromCharCode(41)` if you need one.
<!-- /keep -->

`browse_screenshot()` — capture a full-page PNG of the current browse session and save it under `log/screenshots/`.
Returns the saved path AND the image itself, so you can inspect the page visually (layout, charts, images, anything
text extraction loses).

`send_screenshot(caption: str)` — deliver the most recent `browse_screenshot` to the user over Telegram with an
optional caption. Only available when the user is on Telegram (not voice). Call `browse_screenshot` first so
there is something to send. Use it when the user asks to "see" or "be sent" something visual.

`browse_close()` — shut down the browser and discard the session. Always call this when done. If browse commands
persistently fail, `browse_close` then `browse_open` starts over with a fresh browser.

Worked example. User asks "is there a table for 4 at Gurras saturday at 7pm?":
You: `browse_open("https://www.bokabord.se/restaurang/gurras")`
[snapshot shows date input at [4], guests select at [5], time input at [6], search button at [7]]
You: `browse_act("4=2026-05-02; 5=4; 6=19:00; 7")`
[one batch: date, guests and time filled, then the search button clicked — fresh snapshot shows time slot buttons]
You: `browse_extract`
[full page text with available times]
You: `browse_close`
Then answer the user.

When a site took several attempts to get through, save or update a procedure with the steps that worked, so the
next visit does not have to rediscover them.

### Stop-and-wait
`browse_open`, `browse_act`, `browse_key`, `browse_inspect`, `browse_snapshot`, `browse_wait`, `browse_network`, `browse_click_at`, `browse_eval`, `browse_extract`, `browse_screenshot`, `send_screenshot` return data: emit them and stop generating; the result arrives in the next round.
