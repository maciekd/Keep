// Keep's push relay: a Cloudflare Worker at push.keep-ai.dev that signs a notification with the
// app's key and hands it to Apple (docs/plans/APP_PUSH_PLAN.md). A Keep server cannot hold that
// key; this holds nothing else.
//
// POST /push {"token", "environment", "box"} → {"status", "reason"}, Apple's own answer.
//
// The notification is written here: a caller supplies no visible text, only `box` — ciphertext
// the phone's extension opens with a key this relay never sees. Nothing is stored or logged.

const TOPIC = "com.floxy.Floxy-AI";
const HOSTS = {
  production: "https://api.push.apple.com",
  sandbox: "https://api.sandbox.push.apple.com",
};
const FALLBACK = { title: "Keep", body: "New message" };   // shown when the phone cannot open the box
const MAX_BODY = 8192;
const MAX_BOX = 3000;                   // keeps the payload under Apple's 4 KB
const PROVIDER_TOKEN_SECONDS = 40 * 60; // Apple wants it renewed, but not more often than every 20 minutes
const KEPT_BY_APPLE_SECONDS = 24 * 3600; // how long Apple holds a push for a phone that is off

let provider = { token: "", made: 0 };

export default {
  async fetch(request, env) {
    if (request.method !== "POST" || new URL(request.url).pathname !== "/push") {
      return answer(404, { error: "not found" });
    }
    const text = await request.text();
    if (text.length > MAX_BODY) return answer(413, { error: "too large" });
    let body;
    try {
      body = JSON.parse(text);
    } catch {
      return answer(400, { error: "the body is not JSON" });
    }
    const { token, environment, box } = body ?? {};
    if (typeof token !== "string" || !/^[0-9a-f]{64,200}$/i.test(token)) {
      return answer(400, { error: "token must be a device token in hex" });
    }
    if (!Object.hasOwn(HOSTS, environment)) {
      return answer(400, { error: "environment must be production or sandbox" });
    }
    if (typeof box !== "string" || !box || box.length > MAX_BOX || !/^[A-Za-z0-9+/=]+$/.test(box)) {
      return answer(400, { error: `box must be base64 of at most ${MAX_BOX} characters` });
    }

    const source = request.headers.get("cf-connecting-ip") ?? "unknown";
    const limits = await Promise.all([
      env.PER_TOKEN.limit({ key: token.toLowerCase() }),
      env.PER_SOURCE.limit({ key: source }),
    ]);
    if (limits.some((limit) => !limit.success)) return answer(429, { error: "too many pushes" });

    let response;
    try {
      response = await fetch(`${HOSTS[environment]}/3/device/${token.toLowerCase()}`, {
        method: "POST",
        headers: {
          authorization: `bearer ${await providerToken(env)}`,
          "apns-topic": TOPIC,
          "apns-push-type": "alert",
          "apns-priority": "10",
          "apns-expiration": String(Math.floor(Date.now() / 1000) + KEPT_BY_APPLE_SECONDS),
        },
        body: JSON.stringify({
          aps: { alert: FALLBACK, "mutable-content": 1, sound: "default", "thread-id": "keep" },
          box,
        }),
      });
    } catch (error) {
      return answer(502, { error: `Apple was not reached: ${error.message}` });
    }
    const reason = response.ok ? "" : ((await response.json().catch(() => ({}))).reason ?? "");
    if (response.status === 403) provider.token = "";   // signed anew for the next push
    return answer(200, { status: response.status, reason });
  },
};

// The JWT Apple takes as the proof of who sends: ES256 over the key id and the team id.
async function providerToken(env) {
  const now = Math.floor(Date.now() / 1000);
  if (provider.token && now - provider.made < PROVIDER_TOKEN_SECONDS) return provider.token;
  const pem = env.APNS_KEY.replace(/-----[^-]+-----|\s/g, "");
  const key = await crypto.subtle.importKey(
    "pkcs8", Uint8Array.from(atob(pem), (c) => c.charCodeAt(0)),
    { name: "ECDSA", namedCurve: "P-256" }, false, ["sign"]);
  const unsigned = base64url(JSON.stringify({ alg: "ES256", kid: env.APNS_KEY_ID }))
    + "." + base64url(JSON.stringify({ iss: env.APNS_TEAM_ID, iat: now }));
  const signature = await crypto.subtle.sign(
    { name: "ECDSA", hash: "SHA-256" }, key, new TextEncoder().encode(unsigned));
  provider = { token: `${unsigned}.${base64url(new Uint8Array(signature))}`, made: now };
  return provider.token;
}

function base64url(data) {
  const bytes = typeof data === "string" ? new TextEncoder().encode(data) : data;
  return btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function answer(status, body) {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}
