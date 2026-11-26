# The push relay

The Cloudflare Worker behind `push.keep-ai.dev`: it signs a notification for the Keep app with
the app's APNs key and hands it to Apple. Why it exists and what it may see:
`docs/plans/APP_PUSH_PLAN.md`. Only the maintainer deploys it.

## Deploy

Needs Node, and the `keep-ai.dev` zone in the Cloudflare account `wrangler` is logged in to.

```bash
cd tools/push_relay
npx wrangler login          # once per machine
npx wrangler deploy         # creates push.keep-ai.dev on the first run
```

The three secrets, once (and again when the key is replaced). The key is the `.p8` file's whole
text, piped in so that it is never on the command line — from the maintainer's 1Password item
"Keep push notification certificate", which holds the file:

```bash
op read "op://<vault>/<item>/AuthKey_<key id>.p8" | npx wrangler secret put APNS_KEY
npx wrangler secret put APNS_KEY_ID      # the key's ten characters
npx wrangler secret put APNS_TEAM_ID     # the developer team's id
```

## Try it

With the device token a build of the app run from Xcode prints (`sandbox`; a TestFlight build's
is `production`):

```bash
curl -s https://push.keep-ai.dev/push -d '{"token": "<hex>", "environment": "sandbox", "box": "AAAA"}'
```

`{"status":200,"reason":""}` is Apple taking it: the phone shows "Keep — New message", since no
phone can open that box. Any other status comes with Apple's reason (`BadDeviceToken`: the token
is of the other environment; `InvalidProviderToken`: a secret is wrong).
