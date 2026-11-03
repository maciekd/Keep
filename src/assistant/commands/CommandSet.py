"""The commands one integration adds to the assistant, living in that integration's folder.

A `src/integrations/<folder>/<Name>Commands.py` holds a `CommandSet` subclass with the same
`cmd_<name>` / `describe_<name>` methods `Commands` has for the core commands, and the flags of
its own commands in the four class-level sets. `Commands` builds one instance of every set
`discover()` finds and dispatches to it by command name, so the LLM sees one flat command list
whichever folder a command comes from. A set is registered whether its integration is enabled
or not: a command of a disabled integration answers with that integration's setup hint.
"""
import importlib
import pkgutil
from functools import lru_cache

from common.logging import error, trace

MODULE_SUFFIX = "Commands"


class CommandSet:
    """Base class of an integration's commands; `commands` is the `Commands` hosting it.

    The shared machinery is the host's and is reached at call time — the processor sets the
    channel profile and the request context on the host before every run. Integrations are
    looked up per call (`get_instance()`), never in the constructor: building the sets must
    stay as cheap as building `Commands` itself.

    A `cmd_` method returns "" (or None) when it ran and the reason when it could not — that
    text reaches the LLM as the command's error, and the round is framed as a failure. A
    command with data to hand back does not return it: it hands the work to
    `background_command`, whose return value is the reply.
    """
    # Each names commands of this set only; see the same-named sets on Commands.
    MULTI_PARAM_COMMANDS: frozenset = frozenset()
    SENSITIVE_COMMANDS: frozenset = frozenset()
    PURE_COMMANDS: frozenset = frozenset()
    SELF_EVIDENT_COMMANDS: frozenset = frozenset()
    APPROVAL_NOTES: dict = {}

    def __init__(self, commands):
        self.commands = commands

    @property
    def profile(self):
        """The policy of the channel that asked for the running command."""
        return self.commands._current_profile

    @property
    def context(self):
        """The RequestContext of the running command's request; None for an autonomous run."""
        return self.commands._current_context

    @property
    def command_processor(self):
        return self.commands.command_processor

    def background_command(self, function, params, lane: str | None = None):
        """Hand `function(params)` to a background lane (the host's IO lane unless one of its
        lane constants is given); its return value becomes the command's reply."""
        if lane is None:
            self.commands.background_command(function, params)
        else:
            self.commands.background_command(function, params, lane=lane)

    def fire_and_forget(self, function, params):
        self.commands.fire_and_forget(function, params)


def command_names(owner: type) -> list[str]:
    """The commands a class implements, from its `cmd_` methods. Looked up on the class: a
    getattr on an instance would evaluate its properties too."""
    return [name.removeprefix("cmd_") for name in dir(owner)
            if name.startswith("cmd_") and callable(getattr(owner, name))]


@lru_cache(maxsize=1)
def discover() -> tuple[type[CommandSet], ...]:
    """The command sets of the `integrations` package, found once per process."""
    import integrations
    return find_in(integrations)


def find_in(package) -> tuple[type[CommandSet], ...]:
    """Every CommandSet subclass in a `*Commands` module of `package`, in the package's walk
    order. Only those modules are imported; one that fails to import is logged and left out,
    so a broken folder costs its own commands and nothing else."""
    found: list[type[CommandSet]] = []

    def skipped(fqn):
        error(f"Package {fqn} could not be walked for command sets, its commands are left out")

    for _finder, fqn, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + ".", onerror=skipped):
        if ispkg or not fqn.endswith(MODULE_SUFFIX):
            continue
        try:
            module = importlib.import_module(fqn)
        except Exception as e:
            error(f"Command set module {fqn} could not be imported, its commands are left out: {type(e).__name__}: {e}")
            continue
        for attr in vars(module).values():
            if isinstance(attr, type) and issubclass(attr, CommandSet) and attr is not CommandSet \
                    and attr.__module__ == fqn and attr not in found:
                found.append(attr)
    trace(f"Command sets discovered in {package.__name__}: {', '.join(cls.__name__ for cls in found) or 'none'}")
    return tuple(found)
