"""Finds the console sub-tabs the integration folders bring: the `*Tab` modules under `src/integrations`."""
import importlib
import pkgutil

from common.logging import info
from interfaces.console.tabs.IntegrationTab import IntegrationTab


def discover(package=None) -> list[type[IntegrationTab]]:
    """The IntegrationTab subclasses defined in the `*Tab` modules of `package` (the
    `integrations` package unless given), in walk order. Only those modules are imported — the
    rest of a folder is the assistant's — and one that fails to import is skipped."""
    if package is None:
        import integrations as package
    found: list[type[IntegrationTab]] = []

    def skipped(fqn):
        info(f"Package {fqn} could not be walked for tabs, skipped")

    for _finder, fqn, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + ".", onerror=skipped):
        if ispkg or not fqn.endswith("Tab"):
            continue
        try:
            module = importlib.import_module(fqn)
        except Exception as e:
            info(f"Integration tab module {fqn} could not be imported, skipped: {type(e).__name__}: {e}")
            continue
        for attr in vars(module).values():
            if isinstance(attr, type) and issubclass(attr, IntegrationTab) and attr.__module__ == fqn:
                found.append(attr)
    info(f"Integration tabs discovered: {', '.join(cls.__name__ for cls in found) or 'none'}")
    return found
