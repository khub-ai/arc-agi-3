"""Tier-0 instincts: domain-general priors that propose goals unprompted.

This module introduces a fourth tier beneath the Tier 1 (tools), Tier 2
(synthesized tools), and Tier 3 (compiled Options) hierarchy already
codified in the engine.  **Tier 0 — instincts** are the scaffolding that
shapes *what routines get learned at all*: without a drive to probe
unknowns, no tool or option about their behaviour will ever be
synthesized, because the agent will never collect the observations
required to mint the relevant CausalClaims.

The ARC-AGI-3 benchmark is, in disguise, a test of whether a system can
recover and apply a small set of **human-preference instincts**.  ARC
tasks have (in principle) infinitely many logically-consistent
completions; the "right" one is the one humans prefer, and that
preference is driven by a handful of universal priors — reduce
uncertainty about the unknown, minimise structural difference between
kin structures, avoid repeating actions that produce no new
information.  Coding those priors as first-class instincts is the
honest ARC strategy.

Each instinct is **dual-domain** (works for ARC grid games AND
robotics) and **falsifiable** (ships with a synthetic-world unit
test).  An instinct that only helps one specific game is, by
definition, not an instinct — it is a game adapter.  See
``project_arc_agi3_instincts.md`` in memory for the design spec.

Integration
-----------
The runner calls :meth:`InstinctRegistry.fire_all` once per tick after
perception/mining and before goal synthesis from roles.  Returned
:class:`InstinctProposal`\\s are merged into the goal forest (dedup on
``goal_id``), and action-multiplier proposals are passed through to
the explorer's action-score layer.  Instincts never run the planner
themselves — they only seed *what* the planner considers.

Landing plan (see ``project_arc_agi3_instincts.md``)
----------------------------------------------------
- GAP 23: framework + :class:`ReduceUncertainty` with the ``compare``
  modality only.
- GAP 24 (this landing): adds ``ReduceUncertainty.interact``.  No new
  planner operator was needed — the existing BFS-over-motion-model
  planner already satisfies ``AtPosition`` goals natively, so the
  interact modality simply emits one.  Expected: ls20 L1 tractable
  (agent navigates to unknown entities, explorer then probes actions).
- GAP 25: :class:`MinimizeStructuralDifference` + migration of the
  GAP 21 ``VisualOrientationTrigger`` to be its single-descriptor
  backend.  Makes tr87-level-1 tractable.
- GAP 26: :class:`AvoidRepetition` — factor out the curiosity/novelty
  terms of ``choose_exploration_action`` into an action-level
  instinct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Protocol

from .conditions import AtPosition, Condition, InsideBBox
from .types import Goal, GoalNode, GoalStatus, NodeType, WorldState
from . import goal_forest as _gf


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstinctProposal:
    """One proposal emitted by an :class:`Instinct` on a given tick.

    An instinct may emit zero or more proposals per call.  Each
    proposal either (a) seeds a goal (``goal_id`` + ``condition`` set),
    (b) adjusts action priors (``action_multipliers`` set), or (c)
    both.  Proposals with neither are a no-op and are dropped by the
    registry.

    Parameters
    ----------
    goal_id
        Stable string identifier.  Dedup key — calling ``fire_all``
        twice with an instinct that would re-emit the same goal_id
        does NOT create duplicates.  Convention:
        ``"{instinct_tag}:{modality}:{subject}"`` e.g.
        ``"reduce_uncertainty:compare:e7"``.
    priority
        Goal priority in ``[0.0, 1.0]``.  Compared against adapter-seeded
        and role-derived goals by :func:`goal_forest.candidates_by_priority`.
    condition
        Achievement test.  Evaluated each tick by
        :func:`hypothesis_store.refresh_status`; when ``True`` the goal
        transitions to ``ACHIEVED`` and stops consuming planner cycles.
    rationale
        Human-readable one-line explanation.  Surfaces in telemetry
        and debug dumps — keep it short.
    tags
        Free-form classification tags (e.g. ``{"instinct:reduce_uncertainty",
        "modality:compare"}``).  Intended for telemetry filtering, not
        for behaviour gating.
    action_multipliers
        For action-level instincts like :class:`AvoidRepetition`: a
        mapping ``{action_name: multiplier}`` with values in
        ``[0.0, 1.0]`` applied by the explorer's action-score layer.
        ``None`` for goal-seeding instincts.
    """
    goal_id:            str
    priority:           float
    condition:          Optional[Condition] = None
    rationale:          str                 = ""
    tags:               FrozenSet[str]      = field(default_factory=frozenset)
    action_multipliers: Optional[Dict[str, float]] = None

    def is_goal_proposal(self) -> bool:
        """True if this proposal seeds a goal (has a condition)."""
        return self.condition is not None

    def is_action_proposal(self) -> bool:
        """True if this proposal adjusts action priors."""
        return self.action_multipliers is not None


class Instinct(Protocol):
    """Protocol all instincts satisfy.

    An instinct is a pure function of WorldState: given the current
    world, emit zero or more proposals.  Implementations should be
    side-effect-free with respect to WorldState — the registry is
    responsible for applying proposals, the instinct only suggests.
    """

    name: str

    def fire(self, ws: WorldState) -> List[InstinctProposal]: ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class InstinctRegistry:
    """Holds registered instincts and fires them each tick.

    Usage::

        registry = InstinctRegistry()
        registry.register(ReduceUncertainty())
        # ... each tick:
        proposals = registry.fire_all(ws)
        registry.apply(ws, proposals, step=step)

    Step semantics
    --------------
    ``fire_all`` is a pure query — it does not mutate WorldState.
    ``apply`` is where goal merges and action-multiplier publication
    happen.  Split so that telemetry hooks can observe proposals
    before they land, and so that unit tests can assert on the
    proposal list without needing a full WorldState mutation path.
    """

    def __init__(self) -> None:
        self._instincts:          List[Instinct] = []
        self._action_multipliers: Dict[str, float] = {}

    # -- Registration ------------------------------------------------------

    def register(self, instinct: Instinct) -> None:
        """Register an instinct.  Duplicate names are rejected."""
        existing = {i.name for i in self._instincts}
        if instinct.name in existing:
            raise ValueError(f"instinct already registered: {instinct.name}")
        self._instincts.append(instinct)

    def names(self) -> List[str]:
        return [i.name for i in self._instincts]

    # -- Firing ------------------------------------------------------------

    def fire_all(self, ws: WorldState) -> List[InstinctProposal]:
        """Collect proposals from every registered instinct.

        Pure w.r.t. WorldState: reads only.  Order of returned
        proposals follows registration order so deterministic apply
        semantics hold.
        """
        out: List[InstinctProposal] = []
        for instinct in self._instincts:
            try:
                out.extend(instinct.fire(ws))
            except Exception:
                # An instinct raising must not crash the tick loop.
                # Swallow; telemetry hook will surface this as an
                # "instinct_error" event in a future landing.
                continue
        return out

    # -- Apply -------------------------------------------------------------

    def apply(self,
              ws:        WorldState,
              proposals: List[InstinctProposal],
              *,
              step:      int = 0) -> List[str]:
        """Merge proposals into WorldState.

        Goal proposals become :class:`Goal` objects registered in the
        goal forest (dedup on ``goal_id``; updating an existing goal's
        priority is allowed if the instinct re-fires it at a higher
        value).  Action-multiplier proposals are accumulated into
        ``self._action_multipliers`` for the explorer to consume.

        Returns the list of newly-added goal IDs (for telemetry).
        """
        added: List[str] = []
        # Reset per-tick action multipliers (they are ephemeral by
        # design — an instinct must re-assert them each tick if still
        # valid).
        self._action_multipliers = {}

        for p in proposals:
            if p.is_goal_proposal():
                if p.goal_id in ws.goal_forest.goals:
                    existing = ws.goal_forest.goals[p.goal_id]
                    if p.priority > existing.priority:
                        existing.priority = p.priority
                    continue
                root = GoalNode(
                    id         = f"{p.goal_id}::root",
                    node_type  = NodeType.ATOM,
                    condition  = p.condition,
                    status     = GoalStatus.OPEN,
                    priority   = float(p.priority),
                    source     = f"instinct:{p.goal_id.split(':', 1)[0]}",
                    created_at = step,
                )
                goal = Goal(
                    id         = p.goal_id,
                    root       = root,
                    priority   = float(p.priority),
                    source     = f"instinct:{p.goal_id.split(':', 1)[0]}",
                    created_at = step,
                )
                _gf.add_goal(ws, goal)
                added.append(p.goal_id)

            if p.is_action_proposal():
                for action_name, mult in (p.action_multipliers or {}).items():
                    # Multipliers compose multiplicatively if multiple
                    # instincts target the same action.
                    prev = self._action_multipliers.get(action_name, 1.0)
                    self._action_multipliers[action_name] = prev * float(mult)

        return added

    def action_multiplier(self, action_name: str) -> float:
        """Score multiplier for an action in ``[0.0, 1.0]``.

        Used by the explorer's action-score layer.  Defaults to
        ``1.0`` (no adjustment) for actions no instinct has an
        opinion on.
        """
        return self._action_multipliers.get(action_name, 1.0)


# ---------------------------------------------------------------------------
# Instinct: ReduceUncertainty
# ---------------------------------------------------------------------------


# The minimum claim-coverage below which an entity counts as "unknown"
# and warrants an uncertainty-reduction proposal.  Value picked to
# match :attr:`ExplorerConfig.curiosity_threshold`'s default (0.5).
# Entities with coverage ≥ this threshold already have enough
# hypothesised structure that an instinct-level drive adds no value.
DEFAULT_UNKNOWN_THRESHOLD: float = 0.5


class ReduceUncertainty:
    """Tier-0 instinct: shrink uncertainty about unknowns, cheapest
    modality first.

    Human analogue: when a person encounters something novel, they
    first check whether it resembles something they already know
    ("that's like my other screwdriver"), before picking it up or
    asking someone.  This instinct encodes that hierarchy.

    Robotics analogue: active perception.  A robot presented with a
    new object in its workspace ranks modalities identically —
    passive sensing and comparison to the known-object library come
    before attempted interaction, which comes before escalating to
    the human operator.

    Modalities (cost-ordered)
    -------------------------
    1. **compare** — GAP 23.  If an unknown entity shares at least
       one descriptor (colour, bbox dimensions) with a known entity
       (high claim-coverage), propose a goal whose achievement is
       "claim-coverage of the unknown rises to match the known".
       Free — no motion required; the engine's own comparator tools
       satisfy it.
    2. **observe** — deferred.  Watch the entity for k ticks to see
       if it changes on its own.
    3. **interact** — GAP 24.  When the agent has a committed motion
       model and the entity has a resolvable position, propose an
       ``AtPosition(entity_centroid, agent, tolerance=reach)`` goal
       that the existing BFS planner satisfies natively.  Once the
       agent is there, the explorer picks up and probes ACTION1..N
       via its novelty/info-gain layer; miners then learn the
       causal effect (e.g. ``Touch(e7) → rotation`` on ls20 L1).
    4. **simulate** — deferred.  Requires a forward model.
    5. **ask_observer** — deferred.  LLM escalation, last resort.

    Proposal output
    ---------------
    For each unknown entity ``e`` below ``unknown_threshold`` that
    has a viable modality, emit one :class:`InstinctProposal` with
    ``goal_id = f"reduce_uncertainty:{modality}:{e.id}"`` and priority
    ``(1 - coverage) * modality_weight``.  The ``modality_weight``
    encodes cost — cheaper modalities win when multiple apply.
    """

    name: str = "reduce_uncertainty"

    # Weights applied per modality; cheaper modalities get higher
    # weight so they win dedup when multiple modalities apply to the
    # same unknown.  "compare" is the only one wired up in this
    # landing; others kept in the table for documentation.
    # GAP 24a-2: ``probe_in_place`` sits between ``compare`` (free,
    # wins when it applies) and ``interact`` (reach a target).  It
    # only fires when the agent is *already* inside an unknown
    # entity's bbox — its role is to *hold* the agent there and let
    # the explorer exhaust ACTION1..N until a claim commits and
    # coverage rises, rather than letting a rival ``interact:<other>``
    # goal pull the agent away before anything was learned.  Its
    # weight (0.90) exceeds any ``interact`` priority (max 0.60)
    # across all entities, so the pin is hard while the unknown is
    # still unknown.
    MODALITY_WEIGHTS: Dict[str, float] = {
        "compare":        1.00,
        "probe_in_place": 0.90,
        "observe":        0.80,
        "interact":       0.60,
        "simulate":       0.40,
        "ask_observer":   0.20,
    }

    # Entities whose ids match these reserved names are skipped as
    # interact targets — navigating to the agent itself is vacuous, and
    # placeholder ids like "self" would otherwise force the planner to
    # "reach" itself each tick.
    _AGENT_LIKE_IDS: FrozenSet[str] = frozenset({"agent", "self"})

    def __init__(self,
                 *,
                 unknown_threshold: float = DEFAULT_UNKNOWN_THRESHOLD) -> None:
        self.unknown_threshold = float(unknown_threshold)

    # -- Firing ------------------------------------------------------------

    def fire(self, ws: WorldState) -> List[InstinctProposal]:
        from .explorer import claim_coverage  # local import avoids cycle

        # Partition entities by coverage.
        unknowns: List[str] = []
        knowns:   List[str] = []
        for eid in ws.entities.keys():
            cov = claim_coverage(eid, ws)
            if cov < self.unknown_threshold:
                unknowns.append(eid)
            else:
                knowns.append(eid)

        if not unknowns:
            return []

        proposals: List[InstinctProposal] = []

        # Interact-modality gate: fire only when the agent has a
        # committed motion model.  Without one, BFS navigation cannot
        # plan a path — `_motion_step_tolerance` returns 0.0 and the
        # AtPosition goal would be immediately unreachable.  Defer to
        # a later tick once a MotionModelClaim commits.  Computed once
        # per fire() call since it depends only on ws's committed set.
        motion_available = _gf._motion_step_tolerance(ws) > 0.0

        for uid in unknowns:
            uid_cov = claim_coverage(uid, ws)
            # Hoisted for use by both interact (bbox lookup) and
            # probe_in_place (presence-of-bbox gate).
            target = ws.entities.get(uid)

            # Modality 1: compare
            if knowns:
                similar_kid = self._find_similar_known(ws, uid, knowns)
                if similar_kid is not None:
                    priority = (1.0 - uid_cov) * self.MODALITY_WEIGHTS["compare"]
                    condition = _CoverageAbove(entity_id=uid,
                                               threshold=self.unknown_threshold)
                    proposals.append(InstinctProposal(
                        goal_id   = f"reduce_uncertainty:compare:{uid}",
                        priority  = priority,
                        condition = condition,
                        rationale = (f"unknown {uid} (cov={uid_cov:.2f}) shares "
                                     f"a descriptor with known {similar_kid}; "
                                     f"infer by analogy before interacting."),
                        tags      = frozenset({
                            "instinct:reduce_uncertainty",
                            "modality:compare",
                        }),
                    ))

            # Modality 3: interact
            if motion_available and uid not in self._AGENT_LIKE_IDS:
                pos = _gf._position_for_entity(ws, uid)
                if pos is not None:
                    priority = (1.0 - uid_cov) * self.MODALITY_WEIGHTS["interact"]
                    # Condition selection (GAP 24a-1).  Prefer
                    # ``InsideBBox`` when the entity carries a bbox —
                    # this makes "arrival" mean "agent is actually on
                    # the object's footprint," closing the GAP 24
                    # failure mode where ``AtPosition(centroid, tol)``
                    # flipped to ACHIEVED while the agent still sat
                    # outside the hit-zone.  The planner BFS
                    # terminates on the first frontier state whose
                    # hypothetical agent position falls inside the
                    # bbox — reusing the existing motion-model search
                    # without any new operator.  When no bbox is
                    # available (adapters may report only a point
                    # position for sparse entities), fall back to the
                    # legacy ``AtPosition`` path so this modality
                    # still proposes *something*.
                    bbox   = target.properties.get("bbox") if target else None
                    if bbox is not None:
                        condition: Condition = InsideBBox(
                            entity_id = uid,
                            probe_id  = "agent",
                        )
                    else:
                        tol = _gf._tolerance_for_entity(ws, uid)
                        condition = AtPosition(
                            pos        = tuple(pos),
                            entity_id  = "agent",
                            tolerance  = tol,
                        )
                    proposals.append(InstinctProposal(
                        goal_id   = f"reduce_uncertainty:interact:{uid}",
                        priority  = priority,
                        condition = condition,
                        rationale = (f"unknown {uid} (cov={uid_cov:.2f}) has a "
                                     f"resolvable position and the agent has "
                                     f"a motion model; navigate to it so the "
                                     f"explorer can probe actions and miners "
                                     f"can observe the interaction."),
                        tags      = frozenset({
                            "instinct:reduce_uncertainty",
                            "modality:interact",
                        }),
                    ))

            # Modality 4: probe_in_place (GAP 24a-2).
            #
            # Fires only when the agent is *already* inside the
            # unknown entity's bbox — i.e., the interact modality's
            # navigation phase has succeeded and we're standing on
            # the thing.  Its job is to keep the agent pinned there
            # until coverage rises above threshold, so the explorer
            # (called when the planner can't plan a CoverageAbove
            # goal) exhausts ACTION1..N locally rather than walking
            # off to the next interact target.
            #
            # Condition: ``_CoverageAbove(uid, threshold)`` — the
            # same predicate ``compare`` uses.  It is not plannable
            # by transition model, so the runner falls through to
            # ``choose_exploration_action`` which already prefers
            # novel actions.  When any action at this cell commits
            # a claim involving uid, coverage rises and the goal
            # achieves; uid moves to the ``knowns`` partition on
            # the next tick and no further probe_in_place proposals
            # fire for it.
            #
            # Gate: requires the entity to carry a bbox (InsideBBox
            # is only defined over bboxes) and the agent to be
            # actually inside it *right now*.  When the agent
            # leaves the bbox (plan reroutes, collision bounce), the
            # gate closes — fire() stops emitting this proposal, but
            # the existing goal stays in the forest with its
            # priority until something achieves or preempts it.
            # Re-entry into the same bbox will re-fire the same
            # goal_id; dedup handles the no-op.
            if target is not None and target.properties.get("bbox") is not None \
               and uid not in self._AGENT_LIKE_IDS:
                inside = InsideBBox(entity_id=uid, probe_id="agent").evaluate(ws)
                if inside is True:
                    priority = (1.0 - uid_cov) \
                             * self.MODALITY_WEIGHTS["probe_in_place"]
                    pip_condition = _CoverageAbove(
                        entity_id=uid,
                        threshold=self.unknown_threshold,
                    )
                    proposals.append(InstinctProposal(
                        goal_id   = f"reduce_uncertainty:probe_in_place:{uid}",
                        priority  = priority,
                        condition = pip_condition,
                        rationale = (f"agent is inside bbox of unknown {uid} "
                                     f"(cov={uid_cov:.2f}); hold position and "
                                     f"probe ACTION1..N until a claim commits "
                                     f"and coverage rises above threshold."),
                        tags      = frozenset({
                            "instinct:reduce_uncertainty",
                            "modality:probe_in_place",
                        }),
                    ))

        return proposals

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _find_similar_known(ws:     WorldState,
                            uid:    str,
                            knowns: List[str]) -> Optional[str]:
        """Return a known entity id that shares at least one
        descriptor with the unknown, or ``None``.

        Descriptors compared (this landing): ``colour``, ``bbox``
        dimensions (height, width).  Additional descriptors will be
        added as miners produce them — this deliberately uses only
        properties the perception layer always writes.
        """
        u = ws.entities.get(uid)
        if u is None:
            return None
        u_colour = u.properties.get("colour")
        u_bbox   = u.properties.get("bbox")
        u_dim    = _bbox_dims(u_bbox)

        for kid in knowns:
            k = ws.entities.get(kid)
            if k is None:
                continue
            if u_colour is not None and k.properties.get("colour") == u_colour:
                return kid
            k_dim = _bbox_dims(k.properties.get("bbox"))
            if u_dim is not None and k_dim is not None and u_dim == k_dim:
                return kid
        return None


def _bbox_dims(bbox: Any) -> Optional[tuple]:
    """Return (height, width) tuple for a bbox, or None if malformed.

    Bbox convention (matches engine): ``[row_min, col_min, row_max,
    col_max]`` where integer pixel indices are inclusive.
    """
    if bbox is None:
        return None
    try:
        r0, c0, r1, c1 = (int(x) for x in bbox)
    except (TypeError, ValueError):
        return None
    return (r1 - r0 + 1, c1 - c0 + 1)


# ---------------------------------------------------------------------------
# Supporting condition: CoverageAbove
# ---------------------------------------------------------------------------


from dataclasses import dataclass as _dc


@_dc(frozen=True, eq=False)
class _CoverageAbove(Condition):
    """Predicate: the claim-coverage of an entity is at or above a
    threshold.

    Private-named because it is intended for internal instinct use
    rather than general goal construction; the engine's existing
    conditions already handle positional / resource / role checks.
    """
    entity_id: str
    threshold: float

    def canonical_key(self) -> tuple:
        return ("CoverageAbove", self.entity_id, float(self.threshold))

    def evaluate(self, world: WorldState) -> Optional[bool]:
        from .explorer import claim_coverage
        if self.entity_id not in world.entities:
            # Entity vanished — predicate is undefined.
            return None
        return claim_coverage(self.entity_id, world) >= self.threshold

    def variables(self) -> FrozenSet[str]:
        return frozenset({self.entity_id})
