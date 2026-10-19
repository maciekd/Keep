import importlib
import inspect

from common.ConfigManager import config
from common.logging import info


def create_llm(llm_class_name: str, model: str | None = None, prompt_cache: bool | None = None):
    """Create an LLM instance by class name (e.g. 'OllamaLLM') with optional model override,
    wrapped in a VaultLLM so secrets never reach the provider (a pass-through when the
    vault is disabled or empty).

    prompt_cache is forwarded only to LLMs whose constructor accepts it (e.g. ClaudeLLM);
    other backends silently ignore it.
    """
    from .VaultLLM import VaultLLM  # lazy: VaultLLM imports from this package
    try:
        module = importlib.import_module(f".{llm_class_name}", package="assistant.llm")
    except ModuleNotFoundError:
        raise ValueError(f"Unknown LLM class '{llm_class_name}' — no file assistant/llm/{llm_class_name}.py found")
    cls = getattr(module, llm_class_name)
    kwargs = {}
    if model:
        kwargs['model'] = model
    if prompt_cache is not None and 'prompt_cache' in inspect.signature(cls.__init__).parameters:
        kwargs['prompt_cache'] = prompt_cache
    return VaultLLM(cls(**kwargs))


# Defaults used when the `ai.<tier>` config block is absent — these mirror the
# cloud setup the production server expects, so it runs correctly with no `ai:`
# config at all. The model default is only inherited when the configured class
# also matches the default class (so e.g. `llm: OllamaLLM` won't pick up a Claude model).
# NOTE that if you use the same model for several tiers the kv-cache will work nicely,
# otherwise it will break.
_TIER_DEFAULTS = {
    'fast': ('GeminiLLM', "gemini-3.5-flash"),  # used when user is speaking (voice to voice), neets do be FAST for conversation to feel natural
    'smart': ('ClaudeLLM', 'claude-opus-5-5'),  # used when user is chatting with the AI (over Telegram for example), want the best possible AI here
    'vision': ('ClaudeLLM', 'claude-opus-5-5'),  # used when image processing is needed, overrides smart/fast in that case
    'video': ('GeminiLLM', 'gemini-3.8-flash'),  # describes videos (iMessage/Telegram/email); must be an LLM with native video input — Claude has none
    'autonomous_action': ('ClaudeLLM', 'claude-opus-5-5'),  # used for the heartbeat/autonomous actions
    'memory_curator': ('ClaudeLLM', 'claude-fable-5-1'),  # memory curation: nightly, or more often while the bank is being built up
    'cache_auditor': ('ClaudeLLM', 'claude-sonnet-5-5'),  # decides whether a cached reply may be replayed; a simple verdict, so a cost-efficient model.
                                                        # Sonnet 5.5 replayed prod's latest 104 verdicts 2026-10-19: no refusal, 100 agreed, the 4 others
                                                        # it called UNSAFE where Gemini said SAFE — the harmless direction.
    'photo_indexing': ('GeminiLLM', 'gemini-3.8-flash'),  # indexes every Apple Photos still and video (one call each, thousands on a backfill); needs native video input
    'code_quality': ('ClaudeLLM', 'claude-fable-5-1'),  # model for the nightly Claude Code CLI runs (CodeQualityMaintainer); only the model is used, the class is always the `claude` CLI
}


def content_blocks_contain_images(content) -> bool:
    """True if content is a content-block list holding at least one image block."""
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "image" for b in content
    )


def messages_contain_images(messages: list[dict]) -> bool:
    """True if any message's content is a content-block list holding an image block."""
    return any(content_blocks_contain_images(m.get("content")) for m in messages)


# Tiers `ollama.local_only` leaves alone: code_quality is the `claude` CLI, not an API backend.
_LOCAL_ONLY_EXEMPT = {'code_quality'}
_local_only_logged: set[str] = set()


def local_only_mode() -> bool:
    """True when `ollama.local_only` routes every tier to the Ollama server (`ollama.base_url`)."""
    return bool(config.get('ollama.local_only', False))


def parse_llm_config(tier: str) -> tuple[str, str | None]:
    """Parse config for an LLM tier (e.g. 'fast', 'smart', 'cache_auditor'). Returns (class_name, model_or_None).

    With `ollama.local_only: true` every tier (except code_quality) runs on OllamaLLM regardless of
    its `ai.<tier>` block or cloud default: a tier already on OllamaLLM keeps its configured model,
    any other tier falls back to OllamaLLM's default model — a cloud model name means nothing there."""
    llm, model = _configured_llm(tier)
    if tier not in _LOCAL_ONLY_EXEMPT and local_only_mode():
        if llm != 'OllamaLLM':
            if tier not in _local_only_logged:
                _local_only_logged.add(tier)
                info(f"ollama.local_only: tier '{tier}' runs on OllamaLLM instead of {llm}"
                     f"{f' ({model})' if model else ''}")
            return 'OllamaLLM', None
        return llm, model
    return llm, model


def _configured_llm(tier: str) -> tuple[str, str | None]:
    """The (class, model) the `ai.<tier>` block names, or the tier's cloud default."""
    default_class, default_model = _TIER_DEFAULTS.get(tier, ('OllamaLLM', None))
    try:
        cfg = config.get(f'ai.{tier}')
    except KeyError:
        cfg = None
    if cfg is None:
        return default_class, default_model
    if isinstance(cfg, str):
        return cfg, None
    llm = cfg.get('llm', default_class)
    model = cfg.get('model')
    if model is None and llm == default_class:
        model = default_model
    return llm, model


def tier_api_key_name(tier: str) -> str | None:
    """The api_keys.… entry the LLM configured for *tier* reads, or None when that backend
    needs no key (Ollama). Builds the LLM without initializing it, so it costs nothing and
    works before the key exists — this is how an integration running on a tier (Photos on
    `photo_indexing`) names the key it needs instead of hardcoding a provider."""
    name, model = parse_llm_config(tier)
    key = create_llm(name, model).api_key_name()
    return key
