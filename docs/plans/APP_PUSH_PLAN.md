# Push: what Keep says reaches a phone whose app is closed

Status: a plan, nothing built. Phase 4 of `APP_CHAT_PLAN.md` ("Push while the app is closed").

## Why

The app's chat already holds everything the server says unasked: a `post_message` to `owner`, a
finding of the check, an approval — each is an item of the transcript, and a phone gets it the
next time it connects. Nothing tells the phone that there is something. So today the owner who
has the app still needs Telegram for whatever Keep starts on its own.

After this plan a message Keep sends on its own is a notification on the phone, with its text,
also when the app is not running — and the app is where such messages go by default.

## The obstacle, and the three parts

Apple delivers a notification only for a request signed with the key of the app's team. The
server is open source and self-hosted: an install cannot hold that key. So there is a **relay**,
run by the maintainer, that holds the key and does nothing but sign and forward.

What the relay forwards is **ciphertext**. The phone and its server share a key the relay never
sees; an extension of the app decrypts on the phone. Apple and the relay learn that some Keep
server sent some phone something, and when.

```
Keep server ──box──▶ relay ──signed──▶ Apple ──▶ phone: the extension decrypts ──▶ banner
     ▲                                              │
     └────────── token + key, at registration ──────┘        (LAN / Tailscale, as all of /app)
```

| Part | Where | Size |
|---|---|---|
| The relay | `tools/push_relay/` in this repository — public, so what it does can be read | ~100 lines of JavaScript |
| The server | `src/interfaces/app/`, `src/interfaces/chat/ChatHub.py` | ~300 lines and tests |
| The app | `Keep-Apple`: the app target and a new extension target | ~250 lines of Swift, project settings |

## Where an autonomous message goes

One new config key, at the root:

```yaml
# Where the messages Keep sends on its own are announced (post_message to the owner, what the
# check found, the photos it sends unasked): app — a notification on the phones that have the
# Keep app — or telegram. The other is used while this one reaches nobody. Read live.
notify_channel: app
```

A chat is one of two things. A **list** is the server's own record, read when somebody looks
(the console pane, the app's transcript): it always gets the message, as today. An **alert** is
what makes the owner look: a push to the app, a Telegram message. Exactly one channel alerts.

| `notify_channel` | A phone is registered | Telegram has someone to notify | The message |
|---|---|---|---|
| `app` | yes | — | app: in the list and pushed. Telegram gets nothing |
| `app` | no | yes | Telegram. The app's list has it too, unpushed |
| `telegram` | — | yes | Telegram. The app's list has it too, unpushed |
| `telegram` | yes | no | app: in the list and pushed |
| either | no | no | the lists alone, as today |

"A phone is registered" means: an app installed, pointed at this server, notifications allowed.
That is "the user has the app".

In code:

- `ChatTransport.alerts() -> bool` — whether a message sent here reaches the owner wherever they
  are. Telegram: it has a notification conversation. App: a device is registered. Default false.
- `ChatHub.alert_channel()` — the transport `notify_channel` names when it alerts, else the
  first other one that does, else none. Read at every send.
- `ChatHub._targets(notifications=True)` leaves out a transport for which sending *is* alerting
  (Telegram) when it is not the alert channel. This is the one route `send_message` and the
  unasked media already share.
- `AppTransport` pushes a proactive item only while it is the alert channel.
- `owner_channels()` follows the route (the reactivation telemetry reads it).
- `open_conversation` — a turn the server starts — keeps going to an attended chat first; with
  none attended it opens in the alert channel instead of "the first there is".

The first time a phone registers on an install where Telegram is on, the app's list gets one
code-built notice: messages Keep sends on its own now arrive here, and `notify_channel` says
otherwise. Telegram going quiet must not be a riddle.

## What is pushed

| Item | To | When |
|---|---|---|
| `notice`, `media` — the Notifier's unasked sends | every registered phone | the app is the alert channel, and the send is not `silent` |
| `approval`, `secret` | every registered phone | the app is the alert channel, or a turn of the app's own is running (then the question is asked nowhere else) |
| `assistant`, when it becomes `final` | the phone that sent the turn's last message; every phone for a turn the server started | always |

No presence is tracked. A push to a phone with the app on screen shows nothing — iOS leaves
that to the app, and the app declines — so "is somebody watching" is answered on the phone,
where it is known. The reply's end goes to the asking phone alone so that a reply read on one
device does not ring the others; for that `/app/send` names its phone.

The notification's text is the item's as plain text, cut to what fits (about 2,000 bytes; the
chat has the rest). An approval's is its title. A secret prompt's is its title, never a value.
A photo is its caption or "Photo" — the picture itself is a later step.

## The wire

Two additions to the contract with the apps (`APP_CHAT_PLAN.md`):

**`POST /app/device`** — sent at every launch and whenever the app comes to the front.

| Field | |
|---|---|
| `device` | an id the app made up at its first launch |
| `token` | the APNs device token, hex; `null` forgets the device (notifications were turned off) |
| `environment` | `production` (TestFlight, the App Store) or `sandbox` (a build run from Xcode) |
| `key` | 32 random bytes, base64 — the phone's own, made once, kept in its keychain |
| `name` | "iPhone", "iPad" — for the log |

It also says "this phone has been looked at": the phone's count of unseen pushes, which becomes
the badge, goes back to zero.

**`POST /app/send`** gains an optional `device`, the same id.

The registry is `cache/app_devices.json` (mode 0600 — it holds the keys), capped at a handful of
devices, the oldest registration leaving first. A lost file heals at each phone's next launch.

## Encryption

- The phone makes the key and hands it to its server at registration, over the connection every
  `/app` call uses. Whoever can read that connection can read the chat itself; the key adds
  nothing to what is exposed there. The relay and Apple never see it.
- The box is ChaCha20-Poly1305: a fresh 12-byte nonce, the ciphertext, the 16-byte tag, in that
  order, base64. `cryptography` (already pinned) on the server, CryptoKit's `ChaChaPoly` combined
  form on the phone — the same layout.
- The plaintext is JSON: `title`, `body`, `badge`, `kind`. A contract with the app, pinned by a
  test named so.
- A phone that cannot decrypt (before the first unlock after a reboot, a key the server no
  longer has right) shows what the relay wrote: "Keep — New message".

## The relay

`tools/push_relay/`: a Cloudflare Worker (`worker.js`, `wrangler.toml`, a readme with the
deploy).

**`POST /push`** `{"token", "environment", "box"}` → the APNs status and reason, passed back.

- The relay writes the notification itself — the fixed fallback alert, `mutable-content`, the
  sound, the thread — and attaches `box`. A caller supplies no visible text: the relay cannot be
  made to show a phone anything, only to wake the app's extension with bytes it then fails to
  decrypt.
- `box` is capped (3,000 characters) so the whole payload stays under APNs's 4 KB.
- The topic is the app's bundle id, a constant. The key, key id and team id are Worker secrets.
  The provider token (ES256, WebCrypto) is signed once and reused for 40 minutes.
- No storage, no log of tokens. A limit per device token and per source address.
- The device token is the capability: unguessable, known to the phone and its server.

The server's side is one constant, the relay's address. It should be a name of ours
(`push.keep-ai.dev`), since every released server carries it.

## The server

In `src/interfaces/app/`:

- `Devices.py` — the registry: register, forget, the unseen count, drop a token APNs calls dead
  (`410`, `BadDeviceToken`).
- `Push.py` — the box, the cut to fit, and the sender: a queue and one thread, so no transport
  call waits on the network. A failed send is logged and not retried; the item is in the list.
  Log lines carry the kind, the outcome and a token's first characters — never the text.
- `AppTransport` — decides per the table above at `send_text`, the media sends, the prompts and
  `AppDraft.finalize`; remembers the phone of the last `/app/send`; `alerts()`.
- `routes.py` — `/app/device`, and `device` on `/app/send`.

`ChatHub` gets the route. `config.example.yaml` gets the key. Telemetry: `app_push` with kind
and outcome, no text.

Plain text from an item's HTML: `Destinations.plain_text` or the iMessage transport's
`PlainText`, whichever already does it — not a third.

## The app

- **Capabilities**: Push Notifications on the app; a keychain access group the app and the
  extension share (no App Group — the badge count is the server's, so the two share nothing but
  the key). An entitlements file for iOS: today only the Mac build has one.
- **`Push.swift`**: asks for permission once the chat has been live for the first time;
  registers with APNs; posts `/app/device` at launch, at every return to the front, when the
  token changes and when the server address changes; on coming to the front clears the delivered
  notifications and the badge; declines to present while on screen. The settings panel gets a
  row saying whether notifications are on, leading to the system's switch when they are off.
- **`KeepPush`**, a Notification Service Extension target: reads the key (keychain item readable
  after the first unlock), opens the box, sets title, body and badge. Anything short of a clean
  decrypt delivers the notification as it came.
- `ChatConnection.send` adds `device`.
- **iPhone and iPad only.** The Mac build registers nothing; the extension is left out of it.

## Order of work

Each step ends in something seen on a phone.

1. **The pipe.** The app registers and logs its token (a build from Xcode, sandbox). The relay
   is deployed. A `curl` to the relay shows "Keep — New message" on the phone. This proves the
   one thing only assumed so far: that a Worker reaches APNs.
2. **The text.** Server: registry, box, sender, the route and `notify_channel`. App: the
   extension. A `post_message` to `owner` is a banner with its text, the app closed; with
   `notify_channel: telegram` it goes to Telegram and the app's list has it unpushed.
3. **The rest.** The reply's end, approvals and secrets, the badge, the settings row, the
   first-registration notice, telemetry.
4. **Out.** A server release, then a TestFlight build (`production`). Either order works: an
   app registering at an older server gets a 404 and tries again at its next launch; a server
   with no registered phone behaves as today.

## Tests

`tests/unit/interfaces/app/`:

- the box: opened with the key it gives back the fields; a body of multi-byte characters is cut
  at a character and the payload stays under the cap;
- the registry: a phone's new token replaces its old one, `null` forgets it, a dead token is
  dropped, the count rises per push and falls to zero at `/app/device`;
- the transport over a faked sender: each row of "What is pushed", the `silent` send, the reply
  of a turn sent by one of two phones;
- the routes; the fields of `/app/device` and of the box's plaintext pinned as contracts.

`tests/unit/interfaces/chat/`: each row of the route table, and `open_conversation` with nobody
attended.

The unit conftest refuses the real relay (its refusal joins `test_no_server_relaunch.py`).

The relay and the extension have no unit tests: step 1 and step 2 are their proof, and
`xcrun simctl push` with a box the server's code made exercises the extension in the simulator.

## What only the maintainer can do

- Create the APNs key in the developer portal (a team has two at most); it goes to 1Password.
- A Cloudflare account; deploy the Worker with its three secrets; point `push.keep-ai.dev` at it
  (the zone must be on Cloudflare for a Worker's custom domain — else a `workers.dev` name).
- Run the builds on a phone for steps 1–3, and allow notifications.

## Not yet verified

- A Worker's `fetch` reaching APNs, which requires HTTP/2. A published client for exactly this
  exists (`@fivesheepco/cloudflare-apns2`); step 1 is the proof. If it fails, the relay is the
  same 100 lines on a small host instead.
- One provider token per Worker isolate: APNs refuses tokens renewed too often
  (`TooManyProviderTokenUpdates`). If it shows, the token moves to Workers KV.
- Automatic signing from the command line (`scripts/testflight.sh`) enabling Push on the App ID
  and registering the extension's own.
- The extension reading the shared keychain item on a locked phone, and running for an app that
  was force-quit.

## Later

- The picture of a `media` item in the notification (the extension would fetch it from the
  server, which a phone away from home may not reach).
- Approve and Deny as buttons on the notification.
- The Mac.
- Android: the same registration and box, another relay route (FCM).
