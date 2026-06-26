"""Mechanic learner — accumulates observations, periodically runs
induction, and exposes a current :class:`MechanicCatalog`.

This is the integration surface live game loops plug into.  The
contract is small:

* Per action, the loop calls :meth:`observe_action` with the
  pre-action and post-action agent positions, the action issued,
  the action's expected displacement (from the existing action-
  effect catalog), and the current passability grid.
* Periodically (every ``induction_period`` calls, or on demand), the
  learner runs the inducer over accumulated events and grows the
  catalog when a new mechanic earns support.
* Per consume observation (entity disappeared after a click action),
  the loop calls :meth:`observe_consume` to feed the sibling
  consume-mechanic induction path.
* Persistence lives in two methods, :meth:`to_dict` /
  :meth:`from_dict`, which the live loop's runtime-JSON save/load
  invokes alongside the existing curiosity catalog.

The learner is a thin coordinator; the actual policy lives in
:mod:`mechanic_discovery`.  Keeping the policy independent means
test suites can exercise the inducer directly with hand-built event
lists (as in :mod:`test_mechanic_planner_bp35`) and confirm the
learner's behavior is exactly the composition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple,
)

from cognitive_os.mechanic_catalog import (
    Mechanic,
    MechanicCatalog,
    catalog_from_list,
    catalog_to_list,
)
from cognitive_os.mechanic_discovery import (
    ConsumeEvent,
    MechanicEvent,
    detect_mechanic_event,
    induce_auto_slide_mechanic,
    induce_consume_mechanic,
)


@dataclass
class MechanicLearner:
    """Accumulator for mechanic-discovery events with a current
    committed-mechanics catalog.

    Lifecycle
    ---------
    On construction, the catalog is empty (or loaded from
    persistence).  The live loop feeds in observations via
    :meth:`observe_action` and :meth:`observe_consume`.  Each call
    increments an internal counter; when the counter crosses
    ``induction_period``, :meth:`run_induction` is invoked and the
    counter resets.

    Anti-prior-trap discipline
    --------------------------
    Per :doc:`docs/SPEC_mechanic_discovery_and_planning`, every
    mechanic in the catalog must continue to fit incoming
    observations.  When a committed mechanic's predicate would have
    fired but the observed outcome contradicted it, the learner
    records a contradict event via
    :meth:`MechanicCatalog.increment_contradict`; three strikes
    retire the mechanic (removed from the catalog).  This is the
    check that prevented the bp35 ``arrangement_match`` failure
    mode from being re-introducible.
    """

    catalog:          MechanicCatalog = field(default_factory=MechanicCatalog)
    slide_events:     List[MechanicEvent] = field(default_factory=list)
    consume_events:   List[ConsumeEvent]  = field(default_factory=list)
    induction_period: int                 = 5
    _observations_since_last_induction: int = 0
    contradict_strikes: int               = 3

    # ------------------------------------------------------------------
    # Observation API
    # ------------------------------------------------------------------

    def observe_action(
        self,
        pos_before:            Tuple[int, int],
        pos_after:             Tuple[int, int],
        action_id:             int,
        expected_displacement: Tuple[int, int],
        pgrid,
        frame_shape:           Tuple[int, int],
    ) -> Optional[MechanicEvent]:
        """Feed in a single action observation.  Returns the recorded
        :class:`MechanicEvent` when the observation was a categorical
        surprise (auto-firing mechanic likely involved), or ``None``
        when the action's outcome fits the action-effect model.

        Side effects: appends to ``slide_events``, increments the
        induction counter, validates committed mechanics' falsifiers
        against this observation.
        """
        event = detect_mechanic_event(
            pos_before, pos_after, action_id,
            expected_displacement, pgrid, frame_shape,
        )
        self._observations_since_last_induction += 1
        # Always run validation against committed mechanics, even when
        # the current observation isn't a surprise — a committed
        # auto-slide that *didn't* fire when its precondition was true
        # is a contradiction.
        self._validate_committed_mechanics(
            pos_before, pos_after, action_id,
            expected_displacement, pgrid, frame_shape,
        )
        if event is not None:
            self.slide_events.append(event)
        if self._observations_since_last_induction >= self.induction_period:
            self.run_induction()
            self._observations_since_last_induction = 0
        return event

    def observe_consume(
        self,
        action_id:     int,
        entity_role:   str,
        entity_id:     str,
        click_xy:      Tuple[int, int],
    ) -> None:
        """Feed in a consume observation (entity removed after a
        click action).  The consume-mechanic inducer accumulates
        these and commits when ``min_events`` agree on the same
        (action_id, entity_role) pair."""
        self.consume_events.append(ConsumeEvent(
            action_id   = int(action_id),
            entity_role = str(entity_role),
            entity_id   = str(entity_id),
            click_xy    = (int(click_xy[0]), int(click_xy[1])),
        ))
        self._observations_since_last_induction += 1
        if self._observations_since_last_induction >= self.induction_period:
            self.run_induction()
            self._observations_since_last_induction = 0

    # ------------------------------------------------------------------
    # Induction
    # ------------------------------------------------------------------

    def run_induction(self) -> List[Mechanic]:
        """Run the inducer over all accumulated events.  Commit any
        new mechanics that fit; return the list of newly-committed
        mechanics.

        Re-running induction is idempotent: a mechanic that's already
        in the catalog is not re-added.  Mechanics in the catalog
        have their support count refreshed to reflect the current
        event count.
        """
        committed_now: List[Mechanic] = []

        slide_mech = induce_auto_slide_mechanic(self.slide_events)
        if slide_mech is not None and not self._catalog_has_mechanic(slide_mech):
            self.catalog.add(slide_mech)
            committed_now.append(slide_mech)

        consume_mech = induce_consume_mechanic(self.consume_events)
        if consume_mech is not None and not self._catalog_has_mechanic(consume_mech):
            self.catalog.add(consume_mech)
            committed_now.append(consume_mech)

        return committed_now

    def _catalog_has_mechanic(self, candidate: Mechanic) -> bool:
        """Two mechanics are 'the same' for the catalog's purposes
        if they have the same precondition, action, and effect.
        Credence / support / contradict differ over time but don't
        change identity."""
        for m in self.catalog.all():
            if (m.precondition == candidate.precondition
                    and m.action == candidate.action
                    and m.effect == candidate.effect):
                return True
        return False

    # ------------------------------------------------------------------
    # Falsifier validation
    # ------------------------------------------------------------------

    def _validate_committed_mechanics(
        self,
        pos_before:            Tuple[int, int],
        pos_after:             Tuple[int, int],
        action_id:             int,
        expected_displacement: Tuple[int, int],
        pgrid,
        frame_shape:           Tuple[int, int],
    ) -> None:
        """Walk every committed mechanic.  For each, check whether
        the current observation supports or contradicts it; update
        credence accordingly.  Mechanics with ``contradict`` count
        reaching ``contradict_strikes`` are retired.
        """
        from cognitive_os.mechanic_catalog import EffectKind, ACTION_AUTO
        from cognitive_os.mechanic_discovery import neighborhood_roles

        if not self.catalog.all():
            return
        neighborhood = neighborhood_roles(pos_before, pgrid, frame_shape)
        # Snapshot indices so we can retire mechanics safely.
        retire_indices: List[int] = []
        for idx, m in enumerate(self.catalog.all()):
            # Only validate auto-firing slide-shape mechanics — that's
            # the only family the minimal validator currently handles.
            if m.action != ACTION_AUTO:
                continue
            if m.effect.kind != EffectKind.AGENT_SLIDE:
                continue
            slide_dc = int(m.effect.params.get("dc", 0))
            slide_dr = int(m.effect.params.get("dr", 0))
            # Determine the predicate direction label that matches the
            # mechanic's effect direction.
            label_by_delta = {
                (0, -1): "up", (0, 1): "down",
                (-1, 0): "left", (1, 0): "right",
            }
            predicate_label = label_by_delta.get((slide_dc, slide_dr))
            if predicate_label is None:
                continue
            precondition_held = (
                neighborhood.get(predicate_label) == "walkable"
            )
            obs_dc = pos_after[0] - pos_before[0]
            obs_dr = pos_after[1] - pos_before[1]
            displacement_consistent = (
                (slide_dc == 0 or obs_dc * slide_dc > 0)
                and (slide_dr == 0 or obs_dr * slide_dr > 0)
            )
            # Magnitude check: the slide should produce a magnitude
            # larger than the action's expected step.
            obs_mag = abs(obs_dc) + abs(obs_dr)
            exp_mag = abs(int(expected_displacement[0])) + abs(int(expected_displacement[1]))
            magnitude_consistent = obs_mag > exp_mag

            if precondition_held and displacement_consistent and magnitude_consistent:
                self.catalog.increment_support(idx)
            elif precondition_held and not (displacement_consistent
                                            and magnitude_consistent):
                # Precondition was true but the slide didn't produce
                # the expected effect — a real contradiction.
                self.catalog.increment_contradict(idx)
                refreshed = self.catalog.all()[idx]
                if refreshed.contradict >= self.contradict_strikes:
                    retire_indices.append(idx)
            # When precondition not held, the mechanic doesn't apply
            # to this observation — no credence update.

        # Retire from highest index first so earlier indices stay valid.
        for idx in sorted(retire_indices, reverse=True):
            self.catalog.remove_at(idx)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Snapshot for serialization alongside the existing runtime
        JSON.  Stores the committed catalog plus accumulated raw
        events (so induction can pick up where it left off across
        sessions)."""
        return {
            "catalog":         catalog_to_list(self.catalog),
            "slide_events":    [_slide_event_to_dict(e) for e in self.slide_events],
            "consume_events":  [_consume_event_to_dict(e) for e in self.consume_events],
            "induction_period":   int(self.induction_period),
            "contradict_strikes": int(self.contradict_strikes),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MechanicLearner":
        return cls(
            catalog            = catalog_from_list(d.get("catalog") or []),
            slide_events       = [
                _slide_event_from_dict(e) for e in (d.get("slide_events") or [])
            ],
            consume_events     = [
                _consume_event_from_dict(e) for e in (d.get("consume_events") or [])
            ],
            induction_period   = int(d.get("induction_period", 5)),
            contradict_strikes = int(d.get("contradict_strikes", 3)),
        )

    def bootstrap_consume_from_catalog(
        self,
        curiosity_catalog,
        consume_action_id: int = 6,
    ) -> List[Mechanic]:
        """Seed consume mechanics from the curiosity catalog's
        confirmed-consumable roles, bypassing the slow event-by-event
        induction loop.

        When the live curiosity catalog has ``≥ 2`` recorded
        ``entity_consumed`` events for a role (the existing
        ``is_role_consumable_confirmed`` threshold), the substrate
        already knows that role is consumable.  We commit a
        consume mechanic to the catalog immediately, at a low
        starting credence — the same falsifier discipline applies
        afterward (if the consume stops working, the mechanic gets
        retired via the contradict-strikes path).

        This addresses the "system takes a long time to figure out
        when to use click-consume" failure mode: the mechanic
        catalog gets populated from already-observed evidence at
        first call, instead of waiting for ``min_events`` fresh
        consume events to accumulate.

        Returns the list of newly-committed consume mechanics.
        """
        from cognitive_os.mechanic_catalog import (
            EffectKind, Effect, Precondition, PreconditionKind,
        )
        committed: List[Mechanic] = []
        # Use the catalog's authoritative confirmed-consumable list.
        # On the existing CuriosityCatalog API this is
        # ``consumable_roles()`` (returns roles with >= 2 consume
        # observations).  Per-role observation counts live in
        # ``affordance_consume_count``.
        try:
            confirmed_roles = curiosity_catalog.consumable_roles()
        except Exception:
            return committed
        counts_map = getattr(
            curiosity_catalog, "affordance_consume_count", {},
        )
        for role in confirmed_roles:
            count = int(counts_map.get(str(role), 0))
            candidate = Mechanic(
                name         = f"consume_{role}",
                precondition = Precondition(
                    kind   = PreconditionKind.AGENT_AT_ENTITY,
                    params = {"entity_role": str(role)},
                ),
                action       = int(consume_action_id),
                effect       = Effect(
                    kind   = EffectKind.ENTITY_REMOVE,
                    params = {"entity_role": str(role)},
                ),
                falsifier    = Precondition(
                    kind   = PreconditionKind.AGENT_AT_ENTITY,
                    params = {"entity_role": str(role)},
                ),
                # Bootstrap credence: enough to clear most planner
                # floors, but below "confirmed" so falsifier checks
                # still apply.  Support count seeded with the
                # catalog's observation count so the credence
                # progression is realistic.
                credence     = 0.7,
                support      = int(count),
                contradict   = 0,
            )
            if self._catalog_has_mechanic(candidate):
                continue
            self.catalog.add(candidate)
            committed.append(candidate)
        return committed


# ---------------------------------------------------------------------------
# Internals: event ↔ dict
# ---------------------------------------------------------------------------


def _slide_event_to_dict(e: MechanicEvent) -> Dict[str, Any]:
    return {
        "pos_before":            list(e.pos_before),
        "pos_after":             list(e.pos_after),
        "action_id":             int(e.action_id),
        "expected_displacement": list(e.expected_displacement),
        "neighborhood":          dict(e.neighborhood),
    }


def _slide_event_from_dict(d: Mapping[str, Any]) -> MechanicEvent:
    return MechanicEvent(
        pos_before            = (int(d["pos_before"][0]),  int(d["pos_before"][1])),
        pos_after             = (int(d["pos_after"][0]),   int(d["pos_after"][1])),
        action_id             = int(d["action_id"]),
        expected_displacement = (int(d["expected_displacement"][0]),
                                 int(d["expected_displacement"][1])),
        neighborhood          = dict(d["neighborhood"]),
    )


def _consume_event_to_dict(e: ConsumeEvent) -> Dict[str, Any]:
    return {
        "action_id":   int(e.action_id),
        "entity_role": str(e.entity_role),
        "entity_id":   str(e.entity_id),
        "click_xy":    list(e.click_xy),
    }


def _consume_event_from_dict(d: Mapping[str, Any]) -> ConsumeEvent:
    return ConsumeEvent(
        action_id   = int(d["action_id"]),
        entity_role = str(d["entity_role"]),
        entity_id   = str(d["entity_id"]),
        click_xy    = (int(d["click_xy"][0]), int(d["click_xy"][1])),
    )
