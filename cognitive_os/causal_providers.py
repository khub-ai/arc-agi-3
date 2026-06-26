"""Generalized provider + restore-strategy adapters for the causal-attribution
loop (causal_attribution.py).

The attribution loop is pure and runs against the abstract `Provider`
interface. This module supplies the *binding* layer that is reused across
domains — a game, or a robot — WITHOUT touching the loop:

  - `Observation` — the bundle a domain's sensing returns (conditions,
    relations, agents, fingerprint).
  - `Sandbox` strategies — the part that varies by embodiment:
      * `CopySandbox`   — restore by snapshot/replace state (offline game env
                          deepcopy; a robot SIM rollback to a saved state).
      * `UndoSandbox`   — restore by an Undo capability / inverse actions
                          (online game ACTION7; a robot that reverses its last
                          primitives). No state copy needed.
  - `SandboxProvider` — implements `Provider` from injected callables + a
    sandbox, so any domain plugs in its own sensing and restore.

Nothing here names a game. The ONLY domain-specific things are the callables a
caller injects (how to sense, list actions, project the swept path) and which
sandbox they choose. A robot satisfies the same interface with sensor reads and
an undo/rollback — the loop is identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from .causal_attribution import Relation


@dataclass(frozen=True)
class Observation:
    """One domain sensing read. `relations` are causal_attribution.Relation;
    `fingerprint` is any hashable digest of state used for block confirmation."""
    conditions: frozenset
    relations: tuple
    agents: frozenset
    fingerprint: object


class Sandbox(Protocol):
    """The counterfactual sandbox — apply actions, then return to the prior
    state. This is the part that differs across embodiments."""
    def snapshot(self): ...
    def restore(self, marker) -> None: ...
    def apply(self, action: str) -> None: ...


# Maps a sandbox `kind` to the confidence tier a successful two-arm test earns
# (SPEC_vlm_backward_reasoning §7.1). 'real' rollback confirms about the true
# system; a 'twin' confirms only in a model (bounded by fidelity); 'predict'
# is a mined-dynamics rollforward (weakest).
TIER_BY_KIND = {"real": "confirmed", "twin": "twin_confirmed",
                "predict": "predicted"}


class CopySandbox:
    """Restore by copying/replacing state. For an offline game env (deepcopy =
    REAL system -> kind 'real') or a robot SIM / digital twin (kind 'twin', with
    a `fidelity` note for the fault class). `snapshot_fn` returns an opaque
    state token; `restore_fn` takes it back; `apply_fn` advances the world."""
    def __init__(self, snapshot_fn: Callable[[], object],
                 restore_fn: Callable[[object], None],
                 apply_fn: Callable[[str], None],
                 kind: str = "real", fidelity: Optional[str] = None):
        self._snapshot = snapshot_fn
        self._restore = restore_fn
        self._apply = apply_fn
        self.kind = kind
        self.fidelity = fidelity

    def snapshot(self):
        return self._snapshot()

    def restore(self, marker) -> None:
        self._restore(marker)

    def apply(self, action: str) -> None:
        self._apply(action)


class UndoSandbox:
    """Restore by an Undo capability — no state copy. `apply_fn` advances the
    world; `undo_fn` reverses exactly the last applied action (the game's Undo
    button / ACTION7, or a robot replaying inverse primitives). Tracks depth so
    nested snapshot/restore in the loop reverse cleanly.

    Online-safe but NOT free: every probe step is a real action, so callers cap
    + cache (the loop tries few suspects; verdicts are cacheable by relation
    class). Requires undo to be lossless — calibrate before trusting (the
    capability-probing layer D does this)."""
    def __init__(self, apply_fn: Callable[[str], None],
                 undo_fn: Callable[[], None], kind: str = "real",
                 fidelity: Optional[str] = None):
        self._apply = apply_fn
        self._undo = undo_fn
        self._depth = 0
        self.kind = kind            # real env undo (ARC ACTION7) by default
        self.fidelity = fidelity

    def snapshot(self):
        return self._depth

    def apply(self, action: str) -> None:
        self._apply(action)
        self._depth += 1

    def restore(self, marker) -> None:
        while self._depth > marker:
            self._undo()
            self._depth -= 1


class SandboxProvider:
    """A `Provider` built from injected sensing + a sandbox. Caches one
    `Observation` per state (invalidated on apply/restore) so an expensive
    sensing read happens at most once per state.

    Args:
      observe:      () -> Observation  (the domain's sensing)
      list_actions: () -> list[str]
      sandbox:      a Sandbox (CopySandbox / UndoSandbox / custom)
      project_path: (action) -> list[Relation]  (swept-path suspects; optional)
    """
    def __init__(self, observe: Callable[[], Observation],
                 list_actions: Callable[[], list],
                 sandbox: Sandbox,
                 project_path: Optional[Callable[[str], list]] = None):
        self._observe = observe
        self._list_actions = list_actions
        self._sandbox = sandbox
        self._project_path = project_path or (lambda _a: [])
        self._cache: Optional[Observation] = None

    def _read(self) -> Observation:
        if self._cache is None:
            self._cache = self._observe()
        return self._cache

    # --- Provider interface ---
    def conditions(self) -> frozenset:
        return self._read().conditions

    def relations(self) -> list:
        return list(self._read().relations)

    def agents(self) -> frozenset:
        return self._read().agents

    def fingerprint(self):
        return self._read().fingerprint

    def path_relations(self, action: str) -> list:
        return list(self._project_path(action))

    def actions(self) -> list:
        return list(self._list_actions())

    def snapshot(self):
        return self._sandbox.snapshot()

    def restore(self, marker) -> None:
        self._sandbox.restore(marker)
        self._cache = None

    def apply(self, action: str) -> None:
        self._sandbox.apply(action)
        self._cache = None

    # --- confidence tier (graded by what the sandbox is) ---
    def confirmation_tier(self) -> str:
        return TIER_BY_KIND.get(getattr(self._sandbox, "kind", "real"),
                                "confirmed")

    def confirmation_fidelity(self) -> Optional[str]:
        return getattr(self._sandbox, "fidelity", None)


def to_relation(record) -> Relation:
    """Map a substrate relation record (e.g. relational_kinematics.RelationRecord,
    which has `.kind` and `.entities`) to a causal_attribution.Relation. Used by
    a visual provider's `observe`; kept here so the loop stays decoupled from the
    visual record type."""
    kind = record.kind if hasattr(record, "kind") else record["kind"]
    ents = record.entities if hasattr(record, "entities") else record["entities"]
    return Relation(kind=kind, entities=tuple(ents))
