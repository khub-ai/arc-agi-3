"""Cross-episode knowledge persistence for ARC-AGI-3.

Phase 5c: write the cross-episode primitives out to disk between CLI
invocations so the system accumulates knowledge across runs, not just
within a single ``run_harness`` call.

**What this module persists**

* :class:`cognitive_os.CachedSolution` — action-sequence recordings.
  For ARC-AGI-3 these are the headline cross-episode artefact: the
  exact action sequence that solved a specific level, replayable in
  training mode to skip past already-solved levels.

**What this module deliberately does NOT persist (yet)**

* ``hypotheses`` / ``rules`` — deep, nested :class:`Claim` and
  :class:`Condition` graphs.  The engine does not yet expose a
  canonical serialization surface for these, and rolling one here
  would couple this repo to engine internals that are still evolving
  (Phases 6–7 are expected to rework the hypothesis store).
* ``options`` — the current engine :class:`OptionSynthesiser` is a
  Phase-4 stub that logs candidates but does not construct Option
  instances.  When Phase 7 lands we extend the surface here.
* ``observation_history`` — episode-local; persisting it would blur
  the episode boundary the engine relies on.

The rule of thumb: persist only what has a *stable*, *narrow*,
round-trippable shape; let richer state regenerate from recorded
action sequences plus the engine's own miners.

**Disk layout**

A single JSON file at ``<dir>/knowledge.json``.  Human-readable so
the format can be inspected, diffed, and hand-edited during
debugging.  Versioned via a top-level ``"schema_version"`` integer so
future additions fail loudly rather than silently drop fields.

**Competition-mode purge**

Per the engine's standing convention, training runs load and save;
competition runs load-only or skip persistence entirely.  The harness
exposes both via flags; this module is a dumb data layer that does
what it is told.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from cognitive_os import (
    Action,
    CachedSolution,
    Plan,
    PlanStatus,
    PlannedAction,
    Scope,
    ScopeKind,
    WorldState,
)


_LOG = logging.getLogger("arc_agi_3.persistence")

_SCHEMA_VERSION = 1
_KNOWLEDGE_FILENAME = "knowledge.json"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class LoadReport:
    """Summary of a :func:`load_knowledge` call.

    Returned (rather than mutating-and-forgetting) so harness logs can
    surface how much knowledge actually transferred — the number is a
    useful proxy for "is cross-episode learning working" during
    training sweeps.
    """
    schema_version:       int
    cached_solutions:     int
    skipped_unsupported:  int
    path:                 Path


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _scope_to_dict(scope: Scope) -> Dict[str, Any]:
    # Scope has optional fields that reference frozensets and
    # arbitrary tuples; the only scope kinds CachedSolutions use in
    # Phase 5 are LEVEL / GAME / GLOBAL, which have a bare ``kind``
    # and nothing else.  We still serialize the extra slots if
    # populated so a later phase can round-trip them.
    out: Dict[str, Any] = {"kind": scope.kind.name}
    if scope.position_region is not None:
        out["position_region"] = list(scope.position_region)
    if scope.entity_filter is not None:
        out["entity_filter"] = sorted(scope.entity_filter)
    if scope.time_range is not None:
        out["time_range"] = list(scope.time_range)
    return out


def _scope_from_dict(d: Dict[str, Any]) -> Scope:
    kind = ScopeKind[d.get("kind", "GAME")]
    position_region = tuple(d["position_region"]) if "position_region" in d else None
    entity_filter   = frozenset(d["entity_filter"]) if "entity_filter" in d else None
    time_range      = tuple(d["time_range"])      if "time_range" in d else None
    return Scope(
        kind            = kind,
        position_region = position_region,
        entity_filter   = entity_filter,
        time_range      = time_range,
    )


def _action_to_dict(action: Action) -> Dict[str, Any]:
    return {
        "id":         action.id,
        "name":       action.name,
        "parameters": [[k, _primitive(v)] for k, v in action.parameters],
    }


def _action_from_dict(d: Dict[str, Any]) -> Action:
    params = tuple((str(k), v) for k, v in d.get("parameters", []))
    return Action(id=str(d["id"]), name=str(d["name"]), parameters=params)


def _plan_to_dict(plan: Plan) -> Dict[str, Any]:
    """Shallow plan serialization.

    We keep only the parts a replay needs: goal id, the ordered
    ``Action`` s, and the bookkeeping fields that influence replay
    semantics.  Per-step ``expected_effects`` and ``pre_condition`` s
    are dropped on purpose — they are deep :class:`Claim` /
    :class:`Condition` graphs without a stable serializer yet, and
    replay only needs the actions themselves; the engine's miners
    will re-derive effects as the replay runs.
    """
    return {
        "goal_id":             plan.goal_id,
        "steps":               [_action_to_dict(pa.action) for pa in plan.steps],
        "computed_at":         plan.computed_at,
        "assumptions":         list(plan.assumptions),
        "branch_selections":   dict(plan.branch_selections),
        "status":              plan.status.name,
        "current_step_index":  plan.current_step_index,
    }


def _plan_from_dict(d: Dict[str, Any]) -> Plan:
    steps: List[PlannedAction] = []
    for step_d in d.get("steps", []):
        action = _action_from_dict(step_d)
        # Phase 5c replays reconstruct a minimal PlannedAction: the
        # engine tolerates missing expected_effects / pre_condition
        # at execute time (it falls back to miners and runtime
        # checks).  See :class:`cognitive_os.PlannedAction` docstring.
        steps.append(PlannedAction(
            action                 = action,
            expected_effects       = [],
            depends_on_hypotheses  = [],
            pre_condition          = None,
        ))
    return Plan(
        goal_id             = str(d["goal_id"]),
        steps               = steps,
        computed_at         = int(d.get("computed_at", 0)),
        assumptions         = list(d.get("assumptions", [])),
        branch_selections   = dict(d.get("branch_selections", {})),
        status              = PlanStatus[d.get("status", "COMPLETE")],
        current_step_index  = int(d.get("current_step_index", 0)),
    )


def _cached_solution_to_dict(cs: CachedSolution) -> Dict[str, Any]:
    return {
        "id":               cs.id,
        "task_id":          cs.task_id,
        "plan":             _plan_to_dict(cs.plan),
        "task_parameters":  [[k, _primitive(v)] for k, v in cs.task_parameters],
        "recorded_at":      cs.recorded_at,
        "n_uses":           cs.n_uses,
        "n_successes":      cs.n_successes,
        "deterministic":    cs.deterministic,
        "monitor_level":    cs.monitor_level,
        "scope":            _scope_to_dict(cs.scope),
        "source":           cs.source,
        "rationale":        cs.rationale,
    }


def _cached_solution_from_dict(d: Dict[str, Any]) -> CachedSolution:
    return CachedSolution(
        id              = str(d["id"]),
        task_id         = str(d["task_id"]),
        plan            = _plan_from_dict(d["plan"]),
        task_parameters = tuple((str(k), v) for k, v in d.get("task_parameters", [])),
        recorded_at     = int(d.get("recorded_at", 0)),
        n_uses          = int(d.get("n_uses", 0)),
        n_successes     = int(d.get("n_successes", 0)),
        deterministic   = bool(d.get("deterministic", True)),
        monitor_level   = str(d.get("monitor_level", "low")),
        scope           = _scope_from_dict(d.get("scope", {})),
        source          = str(d.get("source", "postmortem:recording")),
        rationale       = d.get("rationale"),
    )


def _primitive(v: Any) -> Any:
    """Best-effort coercion to JSON-safe primitives for action
    parameters — identical convention to the Mediator's payload
    sanitiser.  An action parameter must be a primitive already in
    practice (the engine does not permit otherwise), but we coerce
    defensively so an out-of-contract value fails loudly on
    deserialization rather than hiding as a ``repr`` string forever."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_primitive(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _primitive(x) for k, x in v.items()}
    return repr(v)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_knowledge(ws: WorldState, directory: os.PathLike | str) -> Path:
    """Write ``ws``'s persistable knowledge to ``<directory>/knowledge.json``.

    Creates ``directory`` if necessary.  Overwrites any existing file
    atomically (write-temp-then-rename) so an interrupted save never
    leaves a corrupt JSON for the next run to choke on.

    Returns the path of the written file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "schema_version":   _SCHEMA_VERSION,
        "cached_solutions": [_cached_solution_to_dict(cs) for cs in ws.cached_solutions.values()],
    }

    target = directory / _KNOWLEDGE_FILENAME
    tmp    = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, target)   # atomic on POSIX and on NTFS (same volume)

    _LOG.info(
        "saved knowledge to %s (%d cached_solutions)",
        target, len(payload["cached_solutions"]),
    )
    return target


def load_knowledge(ws: WorldState, directory: os.PathLike | str) -> LoadReport:
    """Load knowledge from ``<directory>/knowledge.json`` into ``ws``.

    Safe to call on an empty directory — a missing file yields an
    empty :class:`LoadReport`.  Existing entries in ``ws`` are
    preserved; loaded entries merge in by id (loaded wins on id
    collision, matching the "newer knowledge is assumed better"
    heuristic).
    """
    directory = Path(directory)
    target    = directory / _KNOWLEDGE_FILENAME

    if not target.exists():
        return LoadReport(
            schema_version      = _SCHEMA_VERSION,
            cached_solutions    = 0,
            skipped_unsupported = 0,
            path                = target,
        )

    data = json.loads(target.read_text(encoding="utf-8"))

    schema = int(data.get("schema_version", 0))
    if schema > _SCHEMA_VERSION:
        # Newer-on-disk than the code understands — refuse rather
        # than silently losing fields.  The operator either upgrades
        # the code or points the harness at a different dir.
        raise RuntimeError(
            f"knowledge file at {target} has schema_version={schema}, "
            f"but this build supports up to {_SCHEMA_VERSION}."
        )

    loaded_cs        = 0
    skipped          = 0
    for entry in data.get("cached_solutions", []):
        try:
            cs = _cached_solution_from_dict(entry)
        except (KeyError, TypeError, ValueError) as exc:
            # One corrupt entry must not lose the whole file.  Log,
            # count, and move on; the operator can inspect the JSON
            # directly since it's human-readable.
            _LOG.warning("skipping malformed cached_solution entry: %s", exc)
            skipped += 1
            continue
        ws.cached_solutions[cs.id] = cs
        loaded_cs += 1

    report = LoadReport(
        schema_version      = schema,
        cached_solutions    = loaded_cs,
        skipped_unsupported = skipped,
        path                = target,
    )
    _LOG.info(
        "loaded knowledge from %s (%d cached_solutions, %d skipped)",
        target, loaded_cs, skipped,
    )
    return report
