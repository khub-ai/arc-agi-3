"""Side-effect-aware multi-goal planning for the ordered-collection game class.

WHY THIS EXISTS
---------------
The sk48 exploratory loop could DESIGN new sub-goals when stuck (backward-
reasoning protocol + decomposition directive) and backward-reason the steps
for each — but it pursued them ONE AT A TIME via the depends_on chain + the
resource-aware selector. It had no way to compute a SINGLE sequence that
satisfies SEVERAL goals at once (the efficiency: e.g. one staging maneuver
that makes BOTH reds reachable, satisfying the reachability precondition of
two separate collect goals). That conjunctive planner already exists in the
engine — ``cognitive_os.plan_search`` (``enumerate_plans`` + ``SideEffects``
+ side-effect-aware ordering) — but it was never wired into this loop.

This module is the adapter that wires it in for the **ordered-collection-
onto-carrier** game class (sk48's win shape: thread/impale items onto a
carrier in a required order; the HUD draws the order). It is game-agnostic
within that class — nothing here knows "sk48", "arm", "red", or pixel
coordinates. The caller supplies the abstract shape:

  * ``required_order``   the item-roles to collect, in order (from the HUD).
  * ``carrier_initial``  item-roles currently on the carrier (a non-empty
                         carrier is "dirty" — the first collect needs it
                         emptied first; that is the clean-carrier precond).
  * ``reachable_initial``item-roles already reachable by the agent without
                         any staging.
  * ``staging_groups``   enabling maneuvers, each of which makes a SET of
                         item-roles reachable in ONE leg (the side-effect
                         that lets one sequence satisfy several goals).

HOW THE PRECONDITIONS MAP ONTO plan_search
------------------------------------------
``plan_search`` inserts an enabling (service) leg before a primary only when
the primary is non-None but UNAFFORDABLE (a missing resource). A primary
that returns ``None`` aborts the whole ordering with no rescue. So:

  * **Clean-carrier** (fixable by clearing) -> a RESOURCE ``carrier_clean``
    that ``collect[0]`` costs and the ``clear_carrier`` service restores.
    Thus a dirty carrier triggers automatic clear-leg insertion.
  * **Ordering** (collect[i] needs the carrier to already hold the first i
    items) -> ``None`` when the carrier prefix is wrong. No service fixes a
    skipped step, so the wrong-order ordering is correctly abandoned; only
    the HUD order survives. The planner DISCOVERS the order; it is not
    hardcoded.
  * **Reachability** (collect of a staged role needs that role staged) ->
    ``None`` until a staging primary has run. Staging is a PRIMARY (a real
    enabling goal), so it is sequenced ahead by the ordering search, and a
    single staging leg's post-state makes every role in its group reachable
    — satisfying the precondition of every collect of those roles at once.

The output is a ranked ``PlanStack``; the top plan's ``goal_sequence`` is
the computed multi-goal plan (e.g. ``[stage_reds, clear_carrier, collect_0,
collect_1, collect_2]``).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from cognitive_os.resource_aware_selector import GoalId, ResourceKey, ResourceState
from cognitive_os.plan_search import (
    LegSimulation, Plan, PlanStack, SideEffects, enumerate_plans,
)

# Resource keys (engine treats them opaquely).
CLEAN = ResourceKey("carrier_clean")
ACTIONS = ResourceKey("actions")

# Generous action budget — ranking uses total cost, but the budget itself
# never binds for this game class (no consumable fuel).
_ACTION_BUDGET = 999


def _gid(s: str) -> GoalId:
    return GoalId(str(s))


@dataclass(frozen=True)
class CollectionState:
    """Opaque SimState for plan_search. ``carrier`` is the ordered tuple of
    item-roles currently threaded on the carrier; ``staged`` is the set of
    item-roles a staging leg has made reachable."""
    carrier: Tuple[str, ...] = ()
    staged: FrozenSet[str] = frozenset()


@dataclass(frozen=True)
class StagingGroup:
    """An enabling maneuver: in one leg, makes ``makes_reachable`` reachable.
    The whole point of the multi-goal win: ``makes_reachable`` may contain
    several roles, so one leg satisfies the reachability precondition of
    several collects."""
    name: str
    makes_reachable: FrozenSet[str]
    cost: int = 15
    success: float = 0.85


class CollectionSimulator:
    """LegSimulator for the ordered-collection game class."""

    def __init__(
        self,
        required_order: Sequence[str],
        staging_groups: Sequence[StagingGroup],
        base_reachable: Sequence[str],
        collect_cost: int = 4,
        collect_success: float = 0.9,
        clear_cost: int = 7,
        clear_success: float = 0.9,
    ) -> None:
        self.order: List[str] = list(required_order)
        self.groups: Dict[str, StagingGroup] = {g.name: g for g in staging_groups}
        self.base_reachable = set(base_reachable)
        self.collect_cost = collect_cost
        self.collect_success = collect_success
        self.clear_cost = clear_cost
        self.clear_success = clear_success
        self.collect_ids: List[GoalId] = [
            _gid(f"collect_{i}_{role}") for i, role in enumerate(self.order)
        ]
        self.stage_ids: List[GoalId] = [_gid(f"stage_{n}") for n in self.groups]

    # ---- goal partitions for enumerate_plans -----------------------------
    def primaries(self) -> List[GoalId]:
        # Staging maneuvers are real enabling goals (sequenced by the search);
        # the collects are the win-condition goals.
        return self.stage_ids + self.collect_ids

    def services(self) -> List[GoalId]:
        return [_gid("clear_carrier")]

    # ---- helpers ---------------------------------------------------------
    def _reachable(self, state: CollectionState, role: str) -> bool:
        return role in self.base_reachable or role in state.staged

    # ---- the LegSimulator protocol method --------------------------------
    def simulate_leg(self, state: CollectionState, goal_id: GoalId
                     ) -> Optional[LegSimulation]:
        g = str(goal_id)

        # --- clear the carrier (service) ---
        if g == "clear_carrier":
            if not state.carrier:
                # Already clean: return an affordable NO-OP rather than None.
                # The engine's post-leg survival check treats "no service is
                # affordable from post_state" as stranding; this game class
                # has no fuel/stranding concern, so an always-applicable clear
                # keeps that check from false-positiving on an empty carrier.
                return LegSimulation(
                    cost={}, restores={CLEAN: 1}, post_state=state,
                    side_effects=SideEffects(), success_prob=1.0)
            return LegSimulation(
                cost={ACTIONS: self.clear_cost},
                restores={CLEAN: 1},
                post_state=replace(state, carrier=()),
                side_effects=SideEffects(),       # enabling-only, no goal
                success_prob=self.clear_success,
            )

        # --- staging maneuver (primary; multi-role side-effect) ---
        if g.startswith("stage_"):
            grp = self.groups.get(g[len("stage_"):])
            if grp is None:
                return None
            if grp.makes_reachable <= state.staged:
                # Already satisfied -> achieved as a no-op (lets the search
                # skip a redundant staging leg).
                return LegSimulation(
                    cost={}, restores={}, post_state=state,
                    side_effects=SideEffects.achieve(goal_id), success_prob=1.0)
            return LegSimulation(
                cost={ACTIONS: int(grp.cost)},
                restores={},
                post_state=replace(
                    state, staged=frozenset(state.staged | grp.makes_reachable)),
                side_effects=SideEffects.achieve(goal_id),
                success_prob=float(grp.success),
            )

        # --- collect the i-th item (primary; win-condition goal) ---
        if goal_id in self.collect_ids:
            idx = self.collect_ids.index(goal_id)
            role = self.order[idx]
            # Ordering precondition (idx>0): carrier must already hold the
            # required prefix. No service fixes a skipped step -> None aborts
            # this ordering. (idx==0's empty-carrier need is the CLEAN
            # resource + clear_carrier service, NOT a None here, so the
            # clear-leg can be inserted.)
            if idx > 0 and tuple(state.carrier) != tuple(self.order[:idx]):
                return None
            # Reachability precondition: None until a staging leg runs (for
            # roles that need it). Staging is a primary -> sequenced ahead.
            if not self._reachable(state, role):
                return None
            cost: Dict[ResourceKey, int] = {ACTIONS: self.collect_cost}
            if idx == 0:
                cost[CLEAN] = 1  # the first collect needs an empty carrier
            return LegSimulation(
                cost=cost,
                restores={},
                post_state=replace(state, carrier=tuple(state.carrier) + (role,)),
                side_effects=SideEffects.achieve(goal_id),
                success_prob=self.collect_success,
            )

        return None


def plan_ordered_collection(
    *,
    required_order: Sequence[str],
    carrier_initial: Sequence[str] = (),
    reachable_initial: Sequence[str] = (),
    staging_groups: Sequence[StagingGroup] = (),
    generated_at: int = 0,
    max_plans: int = 12,
) -> PlanStack:
    """Compute a ranked stack of side-effect-aware multi-goal plans for an
    ordered-collection task. The top plan's ``goal_sequence`` is the
    computed plan (staging + clears interleaved with the ordered collects)."""
    # Drop staging groups that provide nothing the plan needs: keep a group
    # only if it makes some REQUIRED role reachable that is not already
    # base-reachable. Otherwise staging (a mandatory primary) would be forced
    # even when every item is already reachable.
    needed_roles = set(required_order) - set(reachable_initial)
    staging_groups = [
        g for g in staging_groups
        if frozenset(g.makes_reachable) & needed_roles
    ]
    sim = CollectionSimulator(
        required_order=required_order,
        staging_groups=staging_groups,
        base_reachable=reachable_initial,
    )
    carrier0 = tuple(carrier_initial)
    initial_state = CollectionState(carrier=carrier0, staged=frozenset())
    clean_now = 0 if carrier0 else 1
    initial_resources = [
        ResourceState(key=CLEAN, current=clean_now, full=1),
        ResourceState(key=ACTIONS, current=_ACTION_BUDGET, full=_ACTION_BUDGET),
    ]
    stack = enumerate_plans(
        primaries=sim.primaries(),
        services=sim.services(),
        simulator=sim,
        initial_state=initial_state,
        initial_resources=initial_resources,
        generated_at=generated_at,
        max_plans=max_plans,
    )
    # Re-rank by discovered EFFICIENCY HEURISTICS: convert each plan to the
    # abstract step form and add the heuristic soft-penalty, so churny plans
    # (e.g. an irreversible impale committed before the rest are staged) rank
    # worse. No-op when no heuristics have been discovered yet.
    _apply_heuristic_rerank(stack)
    return stack


def _plan_to_steps(goal_sequence) -> list:
    """Abstract a plan's goal_sequence into efficiency-heuristic PlanSteps:
    stage_* -> {op:stage}, collect_* -> {op:commit} (irreversible), clear ->
    {op:release}. target = the role; lane left None (the abstract planner has
    no lane dimension yet — the lane heuristic activates once it does)."""
    steps = []
    for gid in goal_sequence:
        g = str(gid)
        if g.startswith("stage_"):
            steps.append({"op": "stage", "target": g[len("stage_"):],
                          "lane": None, "reversible": True})
        elif g.startswith("collect_"):
            parts = g.split("_", 2)
            tgt = parts[2] if len(parts) > 2 else g
            steps.append({"op": "commit", "target": tgt, "lane": None,
                          "reversible": False})
        elif g == "clear_carrier":
            steps.append({"op": "release", "target": "*carrier*",
                          "lane": None, "reversible": True})
    return steps


def _apply_heuristic_rerank(stack) -> None:
    try:
        from efficiency_heuristics import load_heuristics, plan_penalty
    except Exception:
        return
    heur = load_heuristics()
    if not heur or not stack.plans:
        return
    def key(p):
        pen = plan_penalty(_plan_to_steps(p.goal_sequence), heur)
        return (not p.feasible, round(p.risk + 0.0, 6),
                sum(p.expected_total_cost.values()) + pen)
    stack.plans.sort(key=key)


# ---------------------------------------------------------------------------
# Producer: build planner inputs from the live world
# ---------------------------------------------------------------------------

def _role_of(name: str) -> str:
    """Reduce an entity name/role to its item-role token. ``block_red`` ->
    ``red``; ``red`` -> ``red``. Keeps the planner role-typed, never tied to
    instance ids."""
    s = str(name)
    for pre in ("block_", "item_", "piece_", "collectable_"):
        if s.startswith(pre):
            return s[len(pre):]
    return s


def build_collection_plan_context(
    world,
    *,
    attachment: Optional[dict] = None,
    required_order: Optional[Sequence[str]] = None,
    reachable_roles: Optional[Sequence[str]] = None,
    staging_cost: int = 15,
) -> Optional[dict]:
    """Assemble ``plan_ordered_collection`` kwargs from the live world, or
    ``None`` if the ordered-collection shape can't be determined.

    Sources (all game-agnostic):
      * **required_order** — the ordered item-roles to collect. Explicit arg,
        else ``world._required_collection_order`` (populated by perception /
        the win-condition reading of the HUD). Without it there is no
        ordered-collection task to plan, so we return None.
      * **carrier_initial** — roles of blocks currently ATTACHED to the agent,
        from the behavioral attachment classifier. A non-empty carrier is
        "dirty" (the first collect needs it cleared).
      * **reachable_roles** — roles already reachable without staging. Explicit
        arg, else ``world._reachable_roles``; defaults to empty (everything
        needs staging) when unknown — conservative, never silently assumes
        reachability.
      * **staging_groups** — one per required role not currently reachable; a
        role appearing in several slots (e.g. red at slots 0 and 2) is made
        reachable by a SINGLE group, so one staging leg satisfies several
        collects (the multi-goal efficiency).
    """
    order = required_order or getattr(world, "_required_collection_order", None)
    if not order:
        return None
    order = [_role_of(r) for r in order]

    att = attachment if attachment is not None else {}
    carrier = sorted(
        _role_of(name) for name, st in att.items() if str(st) == "attached")

    if reachable_roles is None:
        reachable_roles = getattr(world, "_reachable_roles", None)
    base_reach = [_role_of(r) for r in reachable_roles] if reachable_roles is not None else []

    needed = [r for r in dict.fromkeys(order) if r not in base_reach]
    groups = [
        StagingGroup(name=f"{r}s", makes_reachable=frozenset({r}), cost=staging_cost)
        for r in needed
    ]
    return dict(
        required_order=order,
        carrier_initial=carrier,
        reachable_initial=base_reach,
        staging_groups=groups,
    )


# ---------------------------------------------------------------------------
# Human-readable plan-leg labels (for the strategy surface)
# ---------------------------------------------------------------------------

def _leg_label(goal_id: GoalId) -> str:
    g = str(goal_id)
    if g == "clear_carrier":
        return "clear the carrier (un-collect everything so the first item is first)"
    if g.startswith("stage_"):
        return f"staging maneuver '{g[len('stage_'):]}' (make its items reachable in one move)"
    if g.startswith("collect_"):
        parts = g.split("_", 2)
        role = parts[2] if len(parts) > 2 else "?"
        slot = parts[1] if len(parts) > 1 else "?"
        return f"collect a '{role}' (HUD slot {slot})"
    return g


def format_collection_plan(stack: PlanStack, max_alt: int = 2) -> str:
    """Render the top plan(s) for the strategy prompt."""
    if not stack.plans:
        return ("MULTI-GOAL PLAN (side-effect-aware): no feasible plan found "
                "for the current ordered-collection state.")
    lines = ["MULTI-GOAL PLAN (side-effect-aware, computed by goal plan search):",
             "  One sequence that satisfies the ordered-collection goals together "
             "(a single staging leg can satisfy several collects' reachability "
             "preconditions at once — cheaper than one goal at a time)."]
    for pi, plan in enumerate(stack.plans[:max_alt]):
        tag = "BEST" if pi == 0 else f"alt {pi}"
        feas = "feasible" if plan.feasible else "INFEASIBLE"
        lines.append(f"  [{tag}] ({feas}, risk={plan.risk:.2f}, "
                     f"cost={sum(plan.expected_total_cost.values())}):")
        for li, gid in enumerate(plan.goal_sequence, 1):
            lines.append(f"      {li}. {_leg_label(gid)}")
    return "\n".join(lines)
