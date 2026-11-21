# Keep telemetry: what the PostHog events mean

Source of truth: `src/assistant/Telemetry.py`. One install = one `distinct_id` (`telemetry.id` in
the tester's config.yaml). Everything carries `version` (Keep release) and `source`.

| Event | When | Properties worth reading |
|---|---|---|
| `heartbeat` | every server start (`startup: true`) and daily (`startup: false`) | `os_version`, `machine`, `integrations` (enabled list), `content` (transcript still on) |
| `user_turn` | the user spoke | `query` (full text during the first 7 days), `source` (Console, Telegram, Voice…) |
| `assistant_turn` | the assistant answered | `reply` (HTML-ish text with `<b>/<i>/<code>/<a>`), `commands` (names only), `proactive` (sent on its own, source `notifier`) |
| `integration_enabled` / `_disabled` | an integration switched | `integration`, `state` after the restart (`ok`, `warning`, `not_configured`…) |
| `tab_change` | a console tab opened | `tab`, `from`, `dwell` seconds on the tab left |
| `status` | 5 min after start, then hourly if changed | `feeds` (items per running feed), `integrations`, `prompt_kb`, `prompt_sections` |
| `memory_size` | a memory curation ran | `words`, `watch_items`, `procedures` |
| `reactivation_nudge` | a step of the reactivation schedule was spent (`reactivation.enabled`, off by default) | `step` (1–7 daily, 8+ fortnightly), `kind` (`daily`, `backoff`, `fortnightly`), `outcome` (`sent`, `declined` — the model answered NO, `covered` — never asked: the assistant had sent a message of its own within the day), `via` (kinds of the destinations written to: `chat`, `voice`, `group`), `owner_channels` (chats a push could reach), `inactive_hours` |
| `reactivation_return` | the owner's first turn after a day or more of silence | `inactive_hours`, `nudges_sent` (0 = came back unprompted: the base rate), `last_step`, `hours_since_nudge`, `source` |

## Quirks to know before reading a transcript

- **Every server start injects the FTUE prompt as a `user_turn`** whose text begins "The server has
  just started". It is not something the tester typed. Count server starts from `heartbeat
  startup: true`, and count human messages excluding these.
- A `heartbeat startup: true` immediately followed by the FTUE prompt = one restart. Several
  restarts within minutes usually mean the tester is fighting a macOS permission.
- **Installs update themselves at 03:00 local every night** (pull, `sync`, restart) when there is a
  new release, so a start heartbeat around 03:00–03:20 with a new `version` is the nightly update,
  not the tester. Restarts in the daytime are the tester's (or `restart_server` run by the assistant).
- Multi-step replies: one `assistant_turn` can contain several paragraphs separated by newlines,
  one per command round. The `commands` list is for the whole turn.
- `reply` and `query` disappear after the install's 7th day (`content: false` in the heartbeat).
- `status.feeds` empty while an integration is "on" means no feed thread reported a count; do not
  assume indexing ran. `memory_size` absent means the long-term memory was never built.
- `prompt_sections.history` dropping to ~0.5 KB after a restart means the chat history was
  trimmed; the assistant may then re-introduce itself as if meeting the user for the first time.
- Vault values (API keys, passwords, Apple ID) are swapped for placeholders before sending, but
  a 2FA code typed into the chat is not; mask six-digit messages.
- A `reactivation_nudge` carries no text: the message is the `assistant_turn` with `proactive: true`
  of the same moment. The silence is counted from the owner's last turn (any source, the FTUE prompt
  and other server-started turns excluded), so `reactivation_return` is the reliable "came back"
  signal — it needs no guessing at which `user_turn` a human typed.
- Timestamps are UTC. The scripts convert to Europe/Stockholm.

## Reading the FTUE

The FTUE prompt sets the expected path: owner name → iMessage (needs Full Disk Access for the
launching terminal app) → iCloud (Apple ID + password via masked prompt, then 2FA in the
console's Integrations/iCloud tab, Ctrl-R for a fresh code) → Gemini API key (masked prompt) →
Photos (one month indexed, first memory curated, three-year backfill). Judge the install against
that path: which step was reached, how long each took, where the tester went quiet.
