"""Observation -> engine events: the perception->event bridge core.

Engine-clean (P5/P6): turns a per-entity property snapshot delta into typed
:class:`EntityStateChanged` events the miners consume. Adapters supply the
snapshots; this module knows nothing about cells, pixels, sprites, or any
game — it diffs two ``{entity_id: {property: value}}`` mappings.

This is the reusable core the sk48 (and any) adapter's ``observe()`` calls:
perception produces the current per-entity property dict, this differs it
against the previous one, and the resulting events flow into the standard
miner pipeline (``ManipulationTransitionMiner`` -> ``TransitionClaim`` ->
``forward_model.regress`` -> ``plan_to_goal``). See
``docs/SPEC_forward_model.md`` (perception->event bridge, §3 add #4).

Relations (``same_row``, ``clear_column``, ``co_displacement`` …) ride in as
ordinary properties on the relevant entity (e.g. ``{"red": {"clear_of_orange":
True}}``), so the same differ carries the relational preconditions the planner
needs without a separate event type.
"""
from __future__ import annotations

from typing import Any, List, Mapping

from .types import EntityStateChanged, Event

# Sentinel: a property absent from the previous snapshot is a *baseline*, not
# a change (the first observation establishes the value, it does not transit).
_MISSING = object()


def events_from_observation_delta(
    prev: Mapping[str, Mapping[str, Any]],
    curr: Mapping[str, Mapping[str, Any]],
    step: int,
) -> List[Event]:
    """Emit one :class:`EntityStateChanged` per (entity, property) whose value
    differs between ``prev`` and ``curr``.

    - Only entities present in **both** snapshots, and properties present in
      **both** for that entity, are diffed — a newly-observed entity or
      property establishes a baseline silently (no spurious "change").
    - Values must be equality-comparable; unhashable values are fine (only
      ``!=`` is used here, the claim layer handles hashing).
    - Deterministic order: sorted by ``(entity_id, property)`` so the event
      stream — and thus mined claim order — is reproducible.
    """
    events: List[Event] = []
    for eid in sorted(curr.keys()):
        old_props = prev.get(eid)
        if not old_props:
            continue
        new_props = curr[eid]
        for prop in sorted(new_props.keys()):
            old_val = old_props.get(prop, _MISSING)
            if old_val is _MISSING:
                continue
            new_val = new_props[prop]
            if old_val != new_val:
                events.append(EntityStateChanged(
                    step=step, entity_id=eid, property=prop,
                    old=old_val, new=new_val,
                ))
    return events
