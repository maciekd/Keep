"""VaultLLM: the wrapped backend only ever sees `{{secret:<id>}}` placeholders, callers only
ever see real values (VAULT_PLAN §3.6)."""
import pytest
from unittest.mock import Mock

from assistant.llm.AbstractLLM import AbstractLLM
from assistant.llm.VaultLLM import VaultLLM
from common.vault.Vault import Vault

MOBILE = "+46 70-174 06 05"
EMAIL = "m@mlab.test"
PH_MOBILE = "{{secret:marcus-mobile}}"
PH_EMAIL = "{{secret:work-mail}}"


@pytest.fixture
def vault(tmp_path):
    return Vault(path=tmp_path / "vault.bin", passphrase="test")


@pytest.fixture
def stocked(vault):
    vault.add("phone", "Marcus's mobile", MOBILE, id="marcus-mobile")
    vault.add("email", "Work e-mail", EMAIL, id="work-mail")
    return vault


@pytest.fixture
def inner():
    llm = Mock(spec=AbstractLLM)
    llm.name.return_value = "MockLLM (mock-model-1)"
    llm.model = "mock-model-1"
    return llm


@pytest.fixture
def wrapped(inner, stocked):
    return VaultLLM(inner, stocked)


@pytest.fixture
def messages():
    return [{"role": "system", "content": f"The user's e-mail is {EMAIL}."},
            {"role": "user", "content": f"Text {MOBILE} that I'm late"}]


REDACTED = [{"role": "system", "content": f"The user's e-mail is {PH_EMAIL}."},
            {"role": "user", "content": f"Text {PH_MOBILE} that I'm late"}]


@pytest.mark.unit
class TestOutbound:
    def test_str_content_reaches_the_inner_llm_redacted(self, wrapped, inner, messages):
        inner.create_completion.return_value = "ok"
        wrapped.create_completion(messages, thinking=True)
        inner.create_completion.assert_called_once_with(REDACTED, thinking=True)

    def test_caller_messages_are_not_mutated(self, wrapped, inner, messages):
        inner.create_completion.return_value = "ok"
        before = [dict(m) for m in messages]
        wrapped.create_completion(messages)
        assert messages == before

    def test_text_blocks_redacted_image_blocks_shared(self, wrapped, inner):
        image = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}}
        text = {"type": "text", "text": f"Whose number is {MOBILE}?"}
        messages = [{"role": "user", "content": [text, image]}]
        inner.create_completion_with_images.return_value = "ok"

        wrapped.create_completion_with_images(messages, ["img-bytes"])

        sent_messages, sent_images = inner.create_completion_with_images.call_args.args
        assert sent_messages[0]["content"][0] == {"type": "text", "text": f"Whose number is {PH_MOBILE}?"}
        assert sent_messages[0]["content"][1] is image
        assert sent_images == ["img-bytes"]
        assert text["text"] == f"Whose number is {MOBILE}?"

    def test_think_with_video_redacts_both_prompts_and_forwards_the_clip(self, wrapped, inner):
        inner.think_with_video.return_value = f"Send it to {PH_EMAIL}"
        result = wrapped.think_with_video(f"Owner: {EMAIL}", f"Caller: {MOBILE}", b"\x00clip", "video/mp4", fps=2.0)
        inner.think_with_video.assert_called_once_with(
            f"Owner: {PH_EMAIL}", f"Caller: {PH_MOBILE}", b"\x00clip", "video/mp4", fps=2.0)
        assert result == f"Send it to {EMAIL}"

    def test_pre_heat_cache_ships_placeholders_not_values(self, wrapped, inner, messages):
        wrapped.pre_heat_cache(messages)
        inner.pre_heat_cache.assert_called_once_with(REDACTED)

    def test_streaming_sends_redacted_messages(self, wrapped, inner, messages):
        inner.create_streaming_completion.return_value = iter(["ok"])
        list(wrapped.create_streaming_completion(messages, thinking=True))
        inner.create_streaming_completion.assert_called_once_with(REDACTED, thinking=True)


@pytest.mark.unit
class TestInbound:
    def test_create_completion_restores_placeholders(self, wrapped, inner, messages):
        inner.create_completion.return_value = f"Calling {PH_MOBILE} and mailing {PH_EMAIL}."
        assert wrapped.create_completion(messages) == f"Calling {MOBILE} and mailing {EMAIL}."

    def test_create_completion_with_images_restores_placeholders(self, wrapped, inner, messages):
        inner.create_completion_with_images.return_value = f"It reads {PH_MOBILE}"
        assert wrapped.create_completion_with_images(messages, ["img"]) == f"It reads {MOBILE}"

    def test_unknown_placeholder_stays_visible(self, wrapped, inner, messages):
        inner.create_completion.return_value = "Use {{secret:made-up}} here"
        assert wrapped.create_completion(messages) == "Use {{secret:made-up}} here"

    def test_non_string_response_passes_through(self, wrapped, inner, messages):
        inner.create_completion.return_value = None
        assert wrapped.create_completion(messages) is None

    def test_stream_restores_a_placeholder_split_across_chunks(self, wrapped, inner, messages):
        inner.create_streaming_completion.return_value = iter(["Call {{sec", "ret:marcus-mob", "ile}} now"])
        out = list(wrapped.create_streaming_completion(messages))
        assert "".join(out) == f"Call {MOBILE} now"
        assert sum(MOBILE in chunk for chunk in out) == 1, "the value must land in one piece"
        assert all("{{" not in chunk for chunk in out)

    def test_stream_passes_ordinary_text_through_per_chunk(self, wrapped, inner, messages):
        inner.create_streaming_completion.return_value = iter(["Hello ", "there, ", "friend."])
        assert list(wrapped.create_streaming_completion(messages)) == ["Hello ", "there, ", "friend."]

    def test_stream_does_not_start_the_inner_stream_until_iterated(self, wrapped, inner, messages):
        inner.create_streaming_completion.return_value = iter(["x"])
        stream = wrapped.create_streaming_completion(messages)
        inner.create_streaming_completion.assert_not_called()
        next(stream)
        inner.create_streaming_completion.assert_called_once()


@pytest.mark.unit
class TestBaseClassEntryPoints:
    """think()/think_with_messages() are not overridden: the AbstractLLM implementations call
    self.create_completion, which is the wrapper's — so they redact and restore too."""

    def test_think_redacts_both_prompts_and_restores_the_reply(self, wrapped, inner):
        inner.create_completion.return_value = f"Reach the owner at {PH_MOBILE}"
        result = wrapped.think(f"Owner mail: {EMAIL}", f"Owner phone: {MOBILE}")
        inner.create_completion.assert_called_once_with(
            [{"role": "system", "content": f"Owner mail: {PH_EMAIL}"},
             {"role": "user", "content": f"Owner phone: {PH_MOBILE}"}], thinking=False)
        assert result == f"Reach the owner at {MOBILE}"

    def test_think_with_images_goes_through_the_wrapper(self, wrapped, inner):
        inner.create_completion_with_images.return_value = f"Card holder: {PH_EMAIL}"
        result = wrapped.think("sys", f"Is this {MOBILE}?", images=["img"])
        sent_messages, sent_images = inner.create_completion_with_images.call_args.args
        assert sent_messages[1]["content"] == f"Is this {PH_MOBILE}?"
        assert sent_images == ["img"]
        assert result == f"Card holder: {EMAIL}"

@pytest.mark.unit
class TestEmptyVault:
    @pytest.fixture
    def passthrough(self, inner, vault):
        return VaultLLM(inner, vault)

    def test_stream_uses_the_inner_generator_directly(self, passthrough, inner, messages):
        # "{{sec" would be held back by a RestoringStream; unchanged chunks prove there is none.
        chunks = ["a ", "{{sec", " b"]
        inner.create_streaming_completion.return_value = iter(chunks)
        assert list(passthrough.create_streaming_completion(messages, thinking=True)) == chunks
        inner.create_streaming_completion.assert_called_once_with(messages, thinking=True)
        assert inner.create_streaming_completion.call_args.args[0] is messages

    def test_non_streaming_calls_forward_the_very_same_list(self, passthrough, inner, messages):
        inner.create_completion.return_value = f"Echo {MOBILE}"
        assert passthrough.create_completion(messages) == f"Echo {MOBILE}"
        assert inner.create_completion.call_args.args[0] is messages
        passthrough.pre_heat_cache(messages)
        assert inner.pre_heat_cache.call_args.args[0] is messages

    def test_disabled_vault_is_a_pass_through_too(self, inner, tmp_path, messages):
        disabled = Vault(path=tmp_path / "vault.bin", passphrase=None)
        assert not disabled.enabled
        inner.create_completion.return_value = "ok"
        VaultLLM(inner, disabled).create_completion(messages)
        assert inner.create_completion.call_args.args[0] is messages

@pytest.mark.unit
class TestDelegation:
    def test_model_passes_through_getattr(self, wrapped, inner):
        assert wrapped.model == "mock-model-1"
        inner.model = "mock-model-2"
        assert wrapped.model == "mock-model-2"

    def test_private_names_are_never_forwarded(self, wrapped, inner):
        inner._secret_internal = "x"
        with pytest.raises(AttributeError):
            wrapped._secret_internal
        assert not hasattr(wrapped, "__dunder_that_does_not_exist__")

    def test_half_built_instance_raises_instead_of_recursing(self):
        half = VaultLLM.__new__(VaultLLM)
        with pytest.raises(AttributeError):
            half.model

    def test_byte_prefix_tail_preference_is_the_wrapped_backends(self, stocked):
        """AbstractLLM answers False itself, so __getattr__ never forwards the question: a
        backend that wants the tail (Ollama) must still be heard through the wrapper that
        create_llm puts around every backend, or the feature is inert in production."""
        class Backend(AbstractLLM):
            def name(self):
                return "Backend"

            def create_completion(self, messages, thinking=False):
                return ""

            def create_completion_with_images(self, messages, images):
                return ""

            def create_streaming_completion(self, messages, thinking=False):
                yield ""

            def initialize(self):
                return True

        class TailBackend(Backend):
            def prefers_byte_prefix_tail(self):
                return True

        assert VaultLLM(TailBackend(), stocked).prefers_byte_prefix_tail() is True
        assert VaultLLM(Backend(), stocked).prefers_byte_prefix_tail() is False

    def test_refusal_fallback_is_wrapped_in_the_same_vault(self, wrapped, inner, stocked, messages):
        """__getattr__ never reaches refusal_fallback (AbstractLLM defines it), so the
        wrapper must delegate explicitly — and the fallback answers the request this vault
        redacted, so its provider must see placeholders too."""
        fallback = Mock(spec=AbstractLLM)
        fallback.create_completion.return_value = "ok"
        inner.refusal_fallback.return_value = fallback

        result = wrapped.refusal_fallback()

        assert isinstance(result, VaultLLM)
        assert result.redactor.vault is stocked
        result.create_completion(messages)
        fallback.create_completion.assert_called_once_with(REDACTED, thinking=False)

    def test_refusal_fallback_none_stays_none(self, wrapped, inner):
        inner.refusal_fallback.return_value = None
        assert wrapped.refusal_fallback() is None

    def test_last_stop_reason_is_the_inner_llms(self, stocked):
        """A real backend, not a Mock: the property AbstractLLM defines reads
        `_last_stop_reason` off *self*, and __getattr__ refuses private names, so without
        an explicit delegation the wrapper answered None for a reply cut off at max_tokens
        and the memory curator saved the truncated bank anyway."""
        class Backend(AbstractLLM):
            def __init__(self):
                self._last_stop_reason = None

            def create_completion(self, messages, thinking=False):
                self._last_stop_reason = 'max_tokens'
                return "half a memory ba"

            def create_completion_with_images(self, messages, images):
                return ""

            def create_streaming_completion(self, messages, thinking=False):
                yield ""

            def initialize(self):
                return True

            def name(self):
                return "backend"

        wrapped = VaultLLM(Backend(), stocked)
        assert wrapped.last_stop_reason is None
        wrapped.create_completion([{"role": "user", "content": "curate"}])
        assert wrapped.last_stop_reason == 'max_tokens'


@pytest.mark.unit
class TestVaultSelection:
    def test_explicit_vault_is_used_even_when_empty(self, inner, vault, monkeypatch):
        # Vault defines __len__, so an empty vault is falsy — `vault or get_instance()` would
        # silently swap a test vault for the process singleton.
        get_instance = Mock()
        monkeypatch.setattr(Vault, "get_instance", get_instance)
        assert VaultLLM(inner, vault).redactor.vault is vault
        get_instance.assert_not_called()

    def test_no_vault_means_the_singleton(self, inner, monkeypatch):
        get_instance = Mock()
        monkeypatch.setattr(Vault, "get_instance", get_instance)
        assert VaultLLM(inner).redactor.vault is get_instance.return_value
        get_instance.assert_called_once_with()

    def test_new_secrets_are_picked_up_without_rebuilding_the_wrapper(self, inner, vault):
        wrapper = VaultLLM(inner, vault)
        inner.create_completion.return_value = "ok"
        wrapper.create_completion([{"role": "user", "content": f"call {MOBILE}"}])
        assert inner.create_completion.call_args.args[0][0]["content"] == f"call {MOBILE}"

        vault.add("phone", "Marcus's mobile", MOBILE, id="marcus-mobile")
        wrapper.create_completion([{"role": "user", "content": f"call {MOBILE}"}])
        assert inner.create_completion.call_args.args[0][0]["content"] == f"call {PH_MOBILE}"
