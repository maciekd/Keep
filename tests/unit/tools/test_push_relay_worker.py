"""The push relay (tools/push_relay/worker.js), run in node against a stand-in for Apple: what it
refuses before Apple hears of it, that nothing a caller wrote is ever a notification's visible
text, the provider token Apple checks, and its two contracts with the server's side — the body
`interfaces.app.Push` hands it and the answer `Push.post` reads back."""
import base64
import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from common.Paths import AISERVER_DIR
from interfaces.app import Push
from interfaces.app.Devices import Devices
from interfaces.app.Push import Sender

WORKER = AISERVER_DIR / "tools" / "push_relay" / "worker.js"
NOW = 1_795_000_000
KEY_ID, TEAM_ID = "KEYID47KEP", "TEAMKEP47L"
TOKEN = "a1" * 32
PHONE_KEY = base64.b64encode(bytes(range(32))).decode()
BOX = base64.b64encode(b"what only the phone opens").decode()
SMUGGLED = "Marcus, call +46701740605 now"
SOURCE = "192.0.2.7"

HARNESS = r"""
const spec = JSON.parse(process.env.RELAY_SPEC);
let now = spec.now * 1000;
Date.now = () => now;
let apple = {};
let handed = [];
globalThis.fetch = async (url, init) => {
  handed.push({ url: String(url), method: init.method, headers: init.headers, body: init.body });
  if (apple.unreachable) throw new Error("connect timeout");
  const status = apple.status ?? 200;
  return new Response(status === 200 ? "" : JSON.stringify(apple.body ?? {}), { status });
};
const limiter = (blocked) => ({ keys: [], async limit({ key }) { this.keys.push(key); return { success: !blocked.includes(key) }; } });
const out = {};
for (const step of spec.steps) {
  now += (step.advance ?? 0) * 1000;
  apple = step.apple ?? {};
  handed = [];
  const env = { APNS_KEY: spec.key, APNS_KEY_ID: spec.key_id, APNS_TEAM_ID: spec.team_id,
                PER_TOKEN: limiter(step.blocked_tokens ?? []), PER_SOURCE: limiter(step.blocked_sources ?? []) };
  const init = { method: step.method ?? "POST", headers: step.headers ?? {} };
  if (init.method !== "GET") init.body = step.body;
  const response = await worker.fetch(new Request("https://push.example" + (step.path ?? "/push"), init), env);
  out[step.name] = { status: response.status, answer: await response.json(), apple: handed,
                     token_keys: env.PER_TOKEN.keys, source_keys: env.PER_SOURCE.keys };
}
console.log(JSON.stringify(out));
"""


def push(token=TOKEN, environment="sandbox", box=BOX, **more) -> str:
    return json.dumps({"token": token, "environment": environment, "box": box, **more})


def from_the_server(body_text: str) -> str:
    """What `Push.Sender` hands the relay for a notification of *body_text*, as it goes over the wire."""
    devices = Devices()
    devices.register("marcus-iphone-0001", TOKEN.upper(), "production", PHONE_KEY, "Marcus's iPhone")
    posted = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Push, "post", lambda body: posted.append(body) or (200, ""))
        Sender(devices).deliver(devices.get("marcus-iphone-0001"), "notice", "Keep", body_text)
    return json.dumps(posted[0])


REFUSED = {
    "not json": "token=a1&box=AAAA",
    "json null": "null",
    "a short token": push(token="a1" * 31),
    "a token that is not hex": push(token="g1" * 32),
    "a token that is a path": push(token="a1" * 32 + "/../../3/device/" + "b2" * 32),
    "a token that is a number": json.dumps({"token": 1234, "environment": "sandbox", "box": BOX}),
    "no environment": json.dumps({"token": TOKEN, "box": BOX}),
    "an environment every object has": push(environment="constructor"),
    "an environment that is a host": push(environment="https://apns.example"),
    "no box": json.dumps({"token": TOKEN, "environment": "sandbox"}),
    "an empty box": push(box=""),
    "a box of readable words": push(box="The laundry is done."),
    "a box that is an object": json.dumps({"token": TOKEN, "environment": "sandbox", "box": {"title": SMUGGLED}}),
    "a box one character over the cap": push(box="A" * 3001),
}

# In order: the worker keeps its provider token between requests, as it does between two pushes
# of one isolate, so the clock (`advance`, seconds) runs through the steps.
STEPS = (
    [{"name": "a get", "method": "GET"}, {"name": "another path", "path": "/push/extra", "body": push()},
     {"name": "the root", "path": "/", "body": push()},
     {"name": "too large", "body": push(more="x" * 9000)}]
    + [{"name": f"refused: {name}", "body": body} for name, body in REFUSED.items()]
    + [{"name": "limited token", "body": push(token=TOKEN.upper()), "blocked_tokens": [TOKEN]},
       {"name": "limited source", "body": push(token="b2" * 32), "blocked_sources": [SOURCE],
        "headers": {"cf-connecting-ip": SOURCE}},
       {"name": "first", "body": push(token=TOKEN.upper(), environment="production")},
       {"name": "sandbox", "body": push(), "advance": 60},
       {"name": "smuggled", "advance": 60,
        "body": push(title=SMUGGLED, alert={"title": SMUGGLED, "body": SMUGGLED},
                     aps={"alert": {"title": SMUGGLED, "body": SMUGGLED}, "badge": 99})},
       {"name": "19 minutes on", "body": push(), "advance": 17 * 60},
       {"name": "from the server", "server_text": "The laundry is done."},
       {"name": "the longest the server makes", "server_text": "Tvätten är klar — \"äntligen\"\n" * 400},
       {"name": "over an hour on", "body": push(), "advance": 62 * 60},
       {"name": "unregistered", "body": push(), "apple": {"status": 410, "body": {"reason": "Unregistered"}}},
       {"name": "bad token", "body": push(), "apple": {"status": 400, "body": {"reason": "BadDeviceToken"}}},
       {"name": "apple without a reason", "body": push(), "apple": {"status": 500}},
       {"name": "unreachable", "body": push(), "apple": {"unreachable": True}},
       {"name": "before the 403", "body": push()},
       {"name": "provider token refused", "body": push(),
        "apple": {"status": 403, "body": {"reason": "ExpiredProviderToken"}}},
       {"name": "after the 403", "body": push()}])


@pytest.fixture(scope="module")
def signing_key():
    return ec.generate_private_key(ec.SECP256R1())


@pytest.fixture(scope="module")
def ran(signing_key):
    """Every step's outcome: the relay's status and answer, and what Apple was handed."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    pem = signing_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                    serialization.NoEncryption()).decode()
    source = WORKER.read_text()
    assert "export default" in source, "the worker no longer exports its handler the way this test lifts it"
    program = source.replace("export default", "const worker =", 1) + HARNESS
    steps = [{**step, "body": from_the_server(step["server_text"])} if "server_text" in step else step
             for step in STEPS]
    spec = {"now": NOW, "key": pem, "key_id": KEY_ID, "team_id": TEAM_ID, "steps": steps}
    run = subprocess.run([node, "--input-type=module", "-e", program], capture_output=True, text=True, timeout=30,
                         env={"RELAY_SPEC": json.dumps(spec), "PATH": "/usr/bin:/bin"})
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def unpadded(part: str) -> bytes:
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))


def provider_token(step: dict) -> str:
    (request,) = step["apple"]
    scheme, token = request["headers"]["authorization"].split(" ")
    assert scheme.lower() == "bearer"
    return token


def now_at(name: str) -> int:
    """The harness's clock at the step *name*."""
    at = NOW
    for step in STEPS:
        at += step.get("advance", 0)
        if step["name"] == name:
            return at
    raise KeyError(name)


@pytest.mark.unit
class TestWhatTheRelayRefuses:
    @pytest.mark.parametrize("step", ["a get", "another path", "the root"])
    def test_anything_but_a_post_to_push_is_not_found(self, ran, step):
        assert ran[step]["status"] == 404 and ran[step]["apple"] == []

    @pytest.mark.parametrize("name", list(REFUSED))
    def test_a_request_that_is_not_a_push_never_reaches_apple(self, ran, name):
        outcome = ran[f"refused: {name}"]
        assert outcome["status"] == 400 and outcome["apple"] == []

    @pytest.mark.parametrize("name", list(REFUSED))
    def test_a_refused_request_spends_nothing_of_a_phones_allowance(self, ran, name):
        """Otherwise anybody could use up the pushes of a phone whose token they do not even know."""
        outcome = ran[f"refused: {name}"]
        assert outcome["token_keys"] == [] and outcome["source_keys"] == []

    def test_a_body_larger_than_any_push_is_refused_unread(self, ran):
        assert ran["too large"]["status"] == 413 and ran["too large"]["apple"] == []

    def test_a_phone_over_its_allowance_is_not_pushed_however_its_token_is_written(self, ran):
        outcome = ran["limited token"]
        assert outcome["status"] == 429 and outcome["apple"] == []

    def test_a_sender_over_its_allowance_is_not_pushed_whatever_phone_it_names(self, ran):
        outcome = ran["limited source"]
        assert outcome["status"] == 429 and outcome["apple"] == []
        assert outcome["source_keys"] == [SOURCE]


@pytest.mark.unit
class TestWhatAppleIsHanded:
    def test_the_push_goes_to_the_service_that_knows_the_token(self, ran):
        (production,), (sandbox,) = ran["first"]["apple"], ran["sandbox"]["apple"]
        assert production["url"] == f"https://api.push.apple.com/3/device/{TOKEN}"
        assert sandbox["url"] == f"https://api.sandbox.push.apple.com/3/device/{TOKEN}"
        assert production["method"] == sandbox["method"] == "POST"

    def test_the_box_travels_as_it_came_beside_a_text_the_relay_wrote(self, ran):
        (request,) = ran["first"]["apple"]
        payload = json.loads(request["body"])
        assert payload["box"] == BOX
        assert set(payload) == {"aps", "box"}

    def test_the_phones_extension_is_given_the_chance_to_open_the_box(self, ran):
        """Apple runs a notification service extension only for an alert marked mutable."""
        (request,) = ran["first"]["apple"]
        aps = json.loads(request["body"])["aps"]
        assert aps["mutable-content"] == 1 and aps["alert"]["body"]
        assert request["headers"]["apns-push-type"] == "alert"

    def test_nothing_a_caller_wrote_is_shown_on_a_phone(self, ran):
        """The relay signs with the app's key: a caller that could set the alert would have an
        open door to every phone whose token it learns."""
        (request,) = ran["smuggled"]["apple"]
        assert ran["smuggled"]["status"] == 200
        assert SMUGGLED not in json.dumps(request)
        assert json.loads(request["body"]) == json.loads(ran["first"]["apple"][0]["body"])

    def test_apple_is_told_to_give_up_on_a_phone_that_stays_off(self, ran):
        (request,) = ran["first"]["apple"]
        assert now_at("first") < int(request["headers"]["apns-expiration"]) <= now_at("first") + 7 * 86400


@pytest.mark.unit
class TestTheProviderToken:
    def test_it_is_signed_with_the_apps_key_and_says_whose_it_is(self, ran, signing_key):
        head, claims, signature = provider_token(ran["first"]).split(".")
        assert json.loads(unpadded(head)) == {"alg": "ES256", "kid": KEY_ID}
        assert json.loads(unpadded(claims)) == {"iss": TEAM_ID, "iat": now_at("first")}
        raw = unpadded(signature)
        assert len(raw) == 64       # JOSE's r‖s, not the DER an X.509 tool writes
        der = encode_dss_signature(int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big"))
        signing_key.public_key().verify(der, f"{head}.{claims}".encode(), ec.ECDSA(hashes.SHA256()))

    def test_pushes_within_twenty_minutes_share_one(self, ran):
        """Apple answers a token renewed more often with TooManyProviderTokenUpdates."""
        assert now_at("19 minutes on") - now_at("first") == 19 * 60
        assert {provider_token(ran[step]) for step in ("first", "sandbox", "smuggled", "19 minutes on")} \
            == {provider_token(ran["first"])}

    def test_one_older_than_an_hour_is_never_used(self, ran):
        """Apple refuses it as expired."""
        claims = json.loads(unpadded(provider_token(ran["over an hour on"]).split(".")[1]))
        assert now_at("over an hour on") - now_at("first") > 3600
        assert now_at("over an hour on") - claims["iat"] < 3600

    def test_one_apple_refused_is_not_used_again(self, ran):
        refused = provider_token(ran["provider token refused"])
        assert refused == provider_token(ran["before the 403"])
        assert provider_token(ran["after the 403"]) != refused


@pytest.mark.unit
class TestTheContractWithTheServer:
    """Both ends as they are: `interfaces/app/Push.py` and the worker."""

    @pytest.fixture
    def answered(self, monkeypatch):
        """`Push.post` over the answer the relay gave at a step, in place of the network."""
        def by(outcome):
            reply = SimpleNamespace(status_code=outcome["status"], json=lambda: outcome["answer"])
            monkeypatch.setattr(requests, "post", lambda *args, **kwargs: reply)
        return by

    def test_what_the_server_sends_is_a_push_the_relay_takes(self, ran):
        outcome = ran["from the server"]
        assert (outcome["status"], outcome["answer"]["status"]) == (200, 200)
        (request,) = outcome["apple"]
        assert request["url"] == f"https://api.push.apple.com/3/device/{TOKEN}"

    def test_the_longest_notification_the_server_makes_fits_the_relay(self, ran):
        outcome = ran["the longest the server makes"]
        assert outcome["status"] == 200 and len(outcome["apple"]) == 1

    def test_the_whole_notification_fits_what_apple_takes(self, ran):
        (request,) = ran["the longest the server makes"]["apple"]
        assert len(request["body"].encode()) <= 4096

    def test_a_push_apple_took_reads_as_sent(self, ran, answered):
        answered(ran["first"])
        assert Push.post({}) == (200, "")

    @pytest.mark.parametrize("step", ["unregistered", "bad token"])
    def test_a_token_apple_no_longer_knows_takes_its_phone_off_the_servers_list(self, ran, answered, step):
        answered(ran[step])
        devices = Devices()
        devices.register("marcus-iphone-0001", TOKEN, "sandbox", PHONE_KEY)
        assert Sender(devices).deliver(devices.get("marcus-iphone-0001"), "notice", "", "The laundry is done.") == "gone"
        assert devices.all() == []

    @pytest.mark.parametrize("step", ["apple without a reason", "provider token refused"])
    def test_any_other_answer_of_apples_keeps_the_phone(self, ran, answered, step):
        answered(ran[step])
        devices = Devices()
        devices.register("marcus-iphone-0001", TOKEN, "sandbox", PHONE_KEY)
        assert Sender(devices).deliver(devices.get("marcus-iphone-0001"), "notice", "", "The laundry is done.") == "refused"
        assert [device.id for device in devices.all()] == ["marcus-iphone-0001"]

    @pytest.mark.parametrize("step", ["unreachable", "limited token", "refused: a short token", "too large"])
    def test_a_refusal_of_the_relays_own_never_reads_as_apples_verdict_on_the_phone(self, ran, answered, step):
        """Read as Apple's, a 429 or a 502 could cost a phone its registration."""
        assert ran[step]["status"] != 200
        answered(ran[step])
        with pytest.raises(RuntimeError):
            Push.post({})
