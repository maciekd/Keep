"""Several home automation backends presented as one house: the snapshot is the merge of
theirs and a command goes to the backend that owns the name.

Names are the LLM's handle, so the merge keeps each backend's names as they are. A name two
backends both show is suffixed with the backend's name on the later one — "Lamp (Kök) (Home
Assistant)", which the prompt renders as `Lamp [Kök] [Home Assistant]` — while the first
registered keeps the bare name, as it had the house to itself before. Every snapshot rebuilds
the index that routes a name back to its backend (and to the name the backend itself knows);
a command naming something the index has not seen refreshes it once, then answers that no
backend knows the name — never a guess, since the wrong backend could act on a lookalike. A
backend whose snapshot fails keeps its last good one in the merge, so its devices do not read
as gone; only when every backend fails does the snapshot raise, and the prompt section keeps
its last content."""
import time
from typing import Callable, Optional

from common.logging import info, trace
from .HomeAutomation import SECTIONS, Entity, HomeAutomation, HomeSnapshot


class CompositeHome(HomeAutomation):
    """One house out of every ready home automation backend."""

    def __init__(self, backends: list[HomeAutomation]):
        self.backends = list(backends)
        # lowered name as the LLM hands it back → (backend, the backend's own name, the name as shown)
        self._index: dict[str, tuple[HomeAutomation, str, str]] = {}
        self._last: dict[int, HomeSnapshot] = {}                    # id(backend) → its last good snapshot
        self._announced: set[str] = set()                           # the collisions logged already

    @property
    def name(self) -> str:
        return "Home automation (" + " + ".join(b.name for b in self.backends) + ")"

    @property
    def config_section(self) -> str:
        return "home"

    @property
    def state_poll_interval(self) -> float:
        return min(b.state_poll_interval for b in self.backends)

    @staticmethod
    def full_name(entity: Entity) -> str:
        """The name with its area folded in — "Lamp (Kök)" — the form the LLM hands back for
        a `"Lamp [Kök]"` line and every backend resolves."""
        return f"{entity.name} ({entity.area})" if entity.area else entity.name

    def snapshot(self) -> HomeSnapshot:
        merged = HomeSnapshot(taken_at=time.time())
        index: dict[str, tuple[HomeAutomation, str, str]] = {}
        failures = []
        for backend in self.backends:
            snap = self._snapshot_of(backend, failures)
            if snap is None:
                continue
            for section in SECTIONS:
                for entity in getattr(snap, section):
                    getattr(merged, section).append(self._claim(index, backend, entity))
        if failures and len(failures) == len(self.backends):
            raise RuntimeError("; ".join(failures))
        self._index = index
        trace(f"Home automation: merged {len(self.backends)} snapshots into {sum(len(getattr(merged, s)) for s in SECTIONS)} "
              f"entities, {len(index)} names in the index")
        return merged

    def _snapshot_of(self, backend: HomeAutomation, failures: list[str]) -> Optional[HomeSnapshot]:
        try:
            snap = backend.snapshot()
        except Exception as e:
            failures.append(f"{backend.name}: {e}")
            last = self._last.get(id(backend))
            info(f"Home automation: {backend.name}'s snapshot failed ({e}) — "
                 + ("keeping its last good one" if last is not None else "nothing from it yet"))
            return last
        self._last[id(backend)] = snap
        return snap

    def _claim(self, index: dict, backend: HomeAutomation, entity: Entity) -> Entity:
        """Enter the entity's names into the index and return the entity as the prompt shows
        it: unchanged, or suffixed with the backend's name when another backend already owns
        the full name — the suffixed copy is reachable by that shown name alone. The bare name
        of an entity with an area is claimed only when free, so a bare "Lamp" keeps meaning
        what it meant to the backend that had it first."""
        full = self.full_name(entity)
        taken = index.get(full.lower())
        if taken is not None and taken[0] is not backend:
            shown = Entity(f"{full} ({backend.name})", entity.value)
            index[shown.name.lower()] = (backend, full, shown.name)
            if full not in self._announced:
                self._announced.add(full)
                info(f"Home automation: '{full}' is in both {taken[0].name} and {backend.name} — shown as '{shown.name}'")
            return shown
        index.setdefault(full.lower(), (backend, full, full))
        if entity.area:
            index.setdefault(entity.name.lower(), (backend, entity.name, full))
        return entity

    def owner(self, entity: Entity) -> Optional[HomeAutomation]:
        hit = self._index.get(self.full_name(entity).lower())
        return hit[0] if hit else None

    def _lookalikes(self, key: str) -> list[str]:
        """The shown names the key is a part of, in display form — what the LLM should have said."""
        return sorted({HomeAutomation.to_display(shown) for k, (_, _, shown) in self._index.items() if key in k})

    def prune(self, snapshot: HomeSnapshot) -> HomeSnapshot:
        """Each backend's filters on its own entities; an entity the index does not know stays."""
        pruned = HomeSnapshot(taken_at=snapshot.taken_at)
        for section in SECTIONS:
            by_owner: dict[Optional[int], list[Entity]] = {}
            for entity in getattr(snapshot, section):
                owner = self.owner(entity)
                by_owner.setdefault(None if owner is None else id(owner), []).append(entity)
            kept = list(by_owner.get(None, []))
            for backend in self.backends:
                own = by_owner.get(id(backend))
                if own:
                    kept.extend(getattr(backend.prune(HomeSnapshot(**{section: own})), section))
            setattr(pruned, section, kept)
        return pruned

    def _route(self, key: str) -> Optional[tuple[HomeAutomation, str, str]]:
        hit = self._index.get(key)
        if hit is None:
            trace(f"Home automation: '{key}' is not in the name index — taking a fresh snapshot")
            try:
                self.snapshot()
            except Exception as e:
                info(f"Home automation: the refresh for '{key}' failed: {e}")
            hit = self._index.get(key)
        return hit

    def _dispatch(self, verb: str, name: str, call: Callable[[HomeAutomation, str], str]) -> str:
        key = name.strip().lower()
        hit = self._route(key)
        if hit is None:
            lookalikes = self._lookalikes(key) if key else []
            if lookalikes:
                info(f"Home automation: {verb}('{name}') — not a listed name, could be {lookalikes}")
                return f'"{name}" could be {", ".join(lookalikes)} — use the name as listed.'
            info(f"Home automation: {verb}('{name}') — no backend knows the name")
            return f'No device or action named "{name}" in {" or ".join(b.name for b in self.backends)}.'
        backend, own, _ = hit
        trace(f"Home automation: {verb}('{name}') → {backend.name} as '{own}'")
        return call(backend, own)

    def turn_on(self, name: str) -> str:
        return self._dispatch("turn_on", name, lambda b, n: b.turn_on(n))

    def turn_off(self, name: str) -> str:
        return self._dispatch("turn_off", name, lambda b, n: b.turn_off(n))

    def dim(self, name: str, level: int) -> str:
        return self._dispatch("dim", name, lambda b, n: b.dim(n, level))

    def execute(self, name: str) -> str:
        return self._dispatch("execute", name, lambda b, n: b.execute(n))

    def stop(self, name: str) -> str:
        return self._dispatch("stop", name, lambda b, n: b.stop(n))

    def error_logs(self) -> str:
        return "\n\n".join(f"--- {b.name} ---\n{b.error_logs()}" for b in self.backends)
