import importlib
import inspect

from assistant.commands.formats.CommandFormats import CommandFormats
from common.ConfigManager import config
from common.DevelopmentConfig import COMMAND_FORMAT_ENV, command_format_env
from common.logging import error, info


def create_llm(llm_class_name: str, model: str | None = None, prompt_cache: bool | None = None,
               command_format: str | None = None):
    """Create an LLM instance by class name (e.g. 'OllamaLLM') with optional model override,
    wrapped in a VaultLLM so secrets never reach the provider (a pass-through when the
    vault is disabled or empty).

    prompt_cache is forwarded only to LLMs whose constructor accepts it (e.g. ClaudeLLM);
    other backends silently ignore it. command_format — a format name, `keep` / `qwen` /
    `dsml` / `invoke` (see command_format_override) — makes the instance's `command_format()` return that
    class instead of the backend's own; None leaves the backend's choice.
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
    llm = cls(**kwargs)
    if command_format:
        fmt = CommandFormats.by_name(command_format)
        if fmt is None:
            raise ValueError(f"Unknown command format '{command_format}' — one of {', '.join(CommandFormats.BY_NAME)}")
        llm.use_command_format(fmt)
    return VaultLLM(llm)


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
    'memory_curator': ('ClaudeLLM', 'claude-fable-5-2'),  # memory curation: nightly, or more often while the bank is being built up
    'cache_auditor': ('GeminiLLM', 'gemini-3.8-flash'),  # used to decide is a user query is cachable or not, fairly simple decision so a cost efficient AI is good here
    'photo_indexing': ('ClaudeLLM', 'claude-sonnet-5-5'),  # describes each new Apple Photos still as the feed sees it (one call each, a few a day); a Claude model takes stills only — the clips go to the `video` tier's backend
    'photo_backfill': ('ClaudeLLM', 'claude-haiku-5-5'),  # the backlog: thousands of stills on a first run, the sweep behind the cursor, a bulk run — nobody waits for it, so the cheapest model that describes well
    'code_quality': ('ClaudeLLM', 'claude-fable-5-2'),  # model for the nightly Claude Code CLI runs (CodeQualityMaintainer); only the model is used, the class is always the `claude` CLI
}


def content_blocks_contain_images(content) -> bool:
    """True if content is a content-block list holding at least one image block."""
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "image" for b in content
    )


def messages_contain_images(messages: list[dict]) -> bool:
    """True if any message's content is a content-block list holding an image block."""
    return any(content_blocks_contain_images(m.get("content")) for m in messages)


# Tiers `local_backend` leaves alone: code_quality is the `claude` CLI, not an API backend.
_LOCAL_ONLY_EXEMPT = {'code_quality'}
_local_only_logged: set[str] = set()
# The root `local_backend` key names a local server by its config section; each is one LLM class.
LOCAL_BACKENDS = {'ollama': 'OllamaLLM', 'ds4': 'Ds4LLM', 'openai': 'OpenAICompatibleLLM'}
_local_backend_warned: set[str] = set()


def local_only_class() -> str | None:
    """The backend the root `local_backend` key routes every tier to: 'OllamaLLM' for `ollama`,
    'Ds4LLM' for `ds4`, 'OpenAICompatibleLLM' for `openai` (any OpenAI-compatible server — oMLX,
    llama.cpp), None for `none` (the default; `off`, false and empty mean the same — a
    bare `off` is YAML's false). A name no local backend has is logged once and means off, so a
    typo costs the switch, not the start."""
    value = config.get('local_backend', 'none')
    if isinstance(value, bool):
        value = 'ollama' if value else 'none'
    value = str(value or '').strip().lower()
    if value in ('', 'off', 'false', 'none'):
        return None
    if value in LOCAL_BACKENDS:
        return LOCAL_BACKENDS[value]
    if value not in _local_backend_warned:
        _local_backend_warned.add(value)
        error(f"local_backend: '{value}' is not a local backend ({', '.join(LOCAL_BACKENDS)}) — the cloud tiers stay")
    return None


def local_only_mode() -> bool:
    """True when `local_backend` routes every tier to one local backend (see local_only_class)."""
    return local_only_class() is not None


def parse_llm_config(tier: str) -> tuple[str, str | None]:
    """Parse config for an LLM tier (e.g. 'fast', 'smart', 'cache_auditor'). Returns (class_name, model_or_None).

    With `local_backend` set every tier (except code_quality) runs on the one local backend it
    names — `ollama`, `ds4` or `openai` — regardless of its `ai.<tier>` block or cloud default: a tier already on that backend keeps its configured model, any other
    tier falls back to the backend's default model — a cloud model name means nothing there."""
    llm, model = _configured_llm(tier)
    local = local_only_class()
    if tier not in _LOCAL_ONLY_EXEMPT and local is not None:
        if llm != local:
            if tier not in _local_only_logged:
                _local_only_logged.add(tier)
                info(f"local_backend: tier '{tier}' runs on {local} instead of {llm}"
                     f"{f' ({model})' if model else ''}")
            return local, None
        return llm, model
    return llm, model


def command_format_override(tier: str) -> str | None:
    """The command format the `ai.<tier>` block names with `command_format:` — `keep`, `qwen`,
    `dsml` or `invoke`, for an Ollama model that is not a Qwen or to try a syntax —, None when the block
    leaves it to the LLM class. The FLOXY_COMMAND_FORMAT environment variable names one for every
    tier over the blocks (the prompt test runner's --format). A name no format has is logged and
    ignored, so a typo costs the override, not the start."""
    return _command_format_choice(tier)[0]


def command_format_chooser(tier: str) -> str | None:
    """Who named the format `command_format_override(tier)` returns, for the log — the environment
    variable or the tier's config block —, None when neither did."""
    return _command_format_choice(tier)[1]


_env_formats_refused: set[str] = set()   # the bad FLOXY_COMMAND_FORMAT names logged, once each, not once per tier


def _command_format_choice(tier: str) -> tuple[str | None, str | None]:
    if name := command_format_env():
        fmt = CommandFormats.by_name(name)
        if fmt is not None:
            return fmt.name, f"the {COMMAND_FORMAT_ENV} environment variable"
        if name not in _env_formats_refused:
            _env_formats_refused.add(name)
            error(f"{COMMAND_FORMAT_ENV}: no format called '{name}' (one of {', '.join(CommandFormats.BY_NAME)}); "
                  f"the config and the LLM classes decide")
    cfg = _tier_block(tier)
    name = cfg.get('command_format') if isinstance(cfg, dict) else None
    if name is None:
        return None, None
    fmt = CommandFormats.by_name(name)
    if fmt is None:
        error(f"ai.{tier}.command_format: no format called '{name}' (one of "
              f"{', '.join(CommandFormats.BY_NAME)}); the LLM class's own format stands")
        return None, None
    return fmt.name, f"the ai.{tier} command_format override"


def _tier_block(tier: str):
    """The `ai.<tier>` config value as written — a dict, a class name string — or None."""
    try:
        return config.get(f'ai.{tier}')
    except KeyError:
        return None


def _configured_llm(tier: str) -> tuple[str, str | None]:
    """The (class, model) the `ai.<tier>` block names, or the tier's cloud default."""
    default_class, default_model = _TIER_DEFAULTS.get(tier, ('OllamaLLM', None))
    cfg = _tier_block(tier)
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
