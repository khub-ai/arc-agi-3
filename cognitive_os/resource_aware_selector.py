"""Resource-aware goal selector — engine-level domain-agnostic primitive.

See ``docs/SPEC_resource_aware_selector.md`` for the design doc.
This module composes with ``cognitive_os.goal_forest``: the Forest
produces a structurally-filtered actionable goal list (priority-
sorted, dependency-checked, conflict-filtered); this module ranks
that list by resource economics.

Inputs:

  * ``candidates``: Sequence of Goal objects (anything with
    ``.id`` and ``.priority`` attributes).
  * ``profiles``: per-goal ``ResourceProfile`` map supplied by an
    adapter (BFS distance for ARC nav, joint-space pathfinder for
    a manipulator, latency estimate for a workflow agent).
  * ``resources``: current/max levels for each tracked
    ``ResourceState``.
  * ``horizon``: 1 (default) considers only the immediate trip;
    2 forward-simulates whether the next-most-important goal
    would still be pursuable after this trip.

Output:

  * ``ResourceAwareSelection`` describing the chosen goal,
    its score, and a rationale string for telemetry — or ``None``
    when no candidate is viable (caller escalates to a higher-
    level planner / Oracle / supervisor).

The module is intentionally short.  All policy lives in five
ordered steps (reachability → affordability → partition →
parsimony → score) plus an optional horizon-N forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any, FrozenSet, Iterable, Mapping, NewType, Optional, Sequence,
)


ResourceKey = NewType("ResourceKey", str)
GoalId      = NewType("GoalId", str)


# ===========================================================================
# Types
# ===========================================================================


@dataclass(frozen=True)
class ResourceProfile:
    """Adapter-supplied cost/benefit estimate for a goal w.r.t. the
    tracked resources.

    The adapter computes this once per candidate per turn from its
    domain-specific state (BFS distance, action-cost model, joint-
    space planner, etc.).  The engine consumes the profile as opaque
    data — it never reads cells, joints, or pixel coordinates.

    ``cost`` and ``restores`` are non-negative integer counts.
    Resources not relevant to a goal are simply absent from the map.

    ``reachable`` is the adapter's verdict on whether the goal is
    reachable at all (path exists in its action graph); the engine
    respects this without re-deriving it.

    ``notes`` is a free-form diagnostic the adapter may use to
    explain its profile choices in telemetry; never load-bearing.
    """
    cost:        Mapping[ResourceKey, int]
    restores:    Mapping[ResourceKey, int]
    reachable:   bool
    notes:       str = ""


@dataclass
class ResourceState:
    """Current and max for a tracked resource.

    Adapters refresh this each turn from observation (meter, sensor,
    counter).  The engine uses ``current`` for affordability and
    ``full`` for the cap on simulated post-restore levels.

    ``sources`` (the set of GoalIds that restore this resource) is
    derived by the engine at selection time from the ``restores``
    field of every profile; adapters do not populate it.
    """
    key:        ResourceKey
    current:    int
    full:       int
    sources:    FrozenSet[GoalId] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ResourceAwareSelection:
    """Engine output: which goal won and why."""
    goal_id:    GoalId
    rationale:  str
    profile:    ResourceProfile
    score:      float


# ===========================================================================
# Algorithm constants
# ===========================================================================


# Penalty applied to a candidate's score (under horizon > 1) when
# pursuing it would render the next-most-important primary
# unaffordable.  The penalty subtracts ``next_priority *
# _BLOCKING_PENALTY`` from the candidate's base score.  Values
# 0.5–1.0 work in practice: higher values make the selector more
# conservative (avoid trips that block other trips); lower values
# let it greedily take the immediate top priority.
_BLOCKING_PENALTY: float = 0.5


# ===========================================================================
# Internal helpers
# ===========================================================================


def _resource_lookup(
    resources: Sequence[ResourceState],
) -> dict[ResourceKey, ResourceState]:
    return {r.key: r for r in resources}


def _is_affordable(
    profile: ResourceProfile,
    rlu:     Mapping[ResourceKey, ResourceState],
) -> bool:
    """A goal is affordable iff every cost fits within the current
    level of the corresponding resource.  Missing resources are
    treated as unaffordable — adapters that omit a tracked resource
    from their cost map are saying "this goal is free w.r.t. that
    resource", but if they reference an UNKNOWN resource, that's a
    profile error and the goal is rejected."""
    for key, cost in profile.cost.items():
        rs = rlu.get(key)
        if rs is None:
            return False
        if cost > rs.current:
            return False
    return True


def _is_restore(profile: ResourceProfile) -> bool:
    """A restore goal increases at least one tracked resource by a
    positive amount when achieved."""
    return any(amt > 0 for amt in profile.restores.values())


def _best_unblocked_priority(
    restore_profile: ResourceProfile,
    primaries:       Sequence[Any],
    profiles:        Mapping[str, ResourceProfile],
    rlu:             Mapping[ResourceKey, ResourceState],
    affordable_set:  set,
) -> "tuple[float, Optional[str]]":
    """When evaluating a restore goal, its value is the priority of
    the highest-priority CURRENTLY-UNAFFORDABLE primary that this
    restore's pursuit would lift to affordable.  "Pursuit" =
    pay cost, gain restores, cap at full — same simulation as
    :func:`_simulate_resource_state`.  Returns (best_priority,
    best_id) or (0.0, None) when no primary would be un-blocked.

    Also verifies the restore itself is affordable: if the agent
    can't even pay the cost to reach the refuel, the restore
    can't un-block anything."""
    if not _is_affordable(restore_profile, rlu):
        return (0.0, None)
    post = _simulate_resource_state(restore_profile, rlu)
    best_priority: float          = 0.0
    best_id:       Optional[str]  = None
    for p in primaries:
        pid = str(p.id)
        if pid in affordable_set:
            continue  # already affordable; no un-block value
        p_profile = profiles.get(pid)
        if p_profile is None:
            continue
        all_afford = True
        for key, cost in p_profile.cost.items():
            if cost > post.get(key, 0):
                all_afford = False
                break
        if not all_afford:
            continue
        priority = float(getattr(p, "priority", 0.0))
        if priority > best_priority:
            best_priority = priority
            best_id = pid
    return (best_priority, best_id)


def _simulate_resource_state(
    profile: ResourceProfile,
    rlu:     Mapping[ResourceKey, ResourceState],
) -> dict[ResourceKey, int]:
    """Apply a profile's cost (subtract) and restores (add, capped at
    full) to produce the post-pursuit resource state.  Used by the
    horizon-N feasibility check."""
    out: dict[ResourceKey, int] = {k: v.current for k, v in rlu.items()}
    for key, cost in profile.cost.items():
        out[key] = out.get(key, 0) - int(cost)
    for key, gain in profile.restores.items():
        cap = rlu[key].full if key in rlu else 0
        out[key] = min(cap, out.get(key, 0) + int(gain))
    return out


def _next_most_important_primary(
    primaries: Sequence[Any],
    skip_id:   str,
) -> "Optional[Any]":
    """Highest-priority primary whose id != skip_id.  Used by the
    horizon-N check: 'after pursuing skip_id, is the next-most-
    important primary still pursuable?'"""
    best = None
    for p in primaries:
        if str(p.id) == skip_id:
            continue
        if best is None or float(getattr(p, "priority", 0.0)) > float(getattr(best, "priority", 0.0)):
            best = p
    return best


# ===========================================================================
# Selection
# ===========================================================================


def select_resource_aware(
    candidates:        Sequence[Any],
    profiles:          Mapping[str, ResourceProfile],
    resources:         Sequence[ResourceState],
    *,
    horizon:           int   = 1,
    blocking_penalty:  float = _BLOCKING_PENALTY,
    restore_promotion_priority_floor: float = 0.0,
) -> Optional[ResourceAwareSelection]:
    """Pick the highest-value actionable goal under current resource
    constraints, or ``None`` when no candidate is viable.

    See module docstring and ``docs/SPEC_resource_aware_selector.md``
    for the algorithm.  Five ordered steps:

      1. Reachability filter — drop ``profile.reachable=False``.
      2. Affordability check — every ``profile.cost[k]`` ≤
         ``ResourceState[k].current``.
      3. Partition into primary (no restores) vs restore goals.
      4. Restore-parsimony — when any primary is affordable, drop
         restore goals from the candidate pool.
      5. Score and return the top.

    A restore goal selected via the un-block path scores at the
    priority of the primary it lifts to affordable, NOT its own
    raw priority.  This is how parsimony composes with promotion:
    when no primary is affordable, the restore's value is exactly
    the highest-priority primary it un-blocks.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not candidates:
        return None

    rlu = _resource_lookup(resources)

    # ── Step 0: drop trackers ───────────────────────────────────────────
    # Goals with priority <= 0 are tracker / status-indicator goals
    # (e.g. BudgetPressureActive) — they exist to surface state, not
    # to be selected as navigation targets.  Removing them at the
    # input keeps the parsimony rule from being misled (a tracker
    # with empty cost would otherwise count as a vacuously-affordable
    # primary, triggering refuel-drop even when no real primary is
    # affordable).
    candidates = [
        g for g in candidates
        if float(getattr(g, "priority", 0.0)) > 0.0
    ]
    if not candidates:
        return None

    # ── Step 1: reachability filter ─────────────────────────────────────
    reachable_candidates = []
    for g in candidates:
        gid = str(g.id)
        profile = profiles.get(gid)
        if profile is None:
            continue
        if not profile.reachable:
            continue
        reachable_candidates.append(g)
    if not reachable_candidates:
        return None

    # ── Step 2: affordability ───────────────────────────────────────────
    affordable_set: set = set()
    for g in reachable_candidates:
        gid = str(g.id)
        profile = profiles[gid]
        if _is_affordable(profile, rlu):
            affordable_set.add(gid)

    # ── Step 3: partition ───────────────────────────────────────────────
    primaries = [g for g in reachable_candidates
                 if not _is_restore(profiles[str(g.id)])]
    restores  = [g for g in reachable_candidates
                 if _is_restore(profiles[str(g.id)])]

    # ── Step 4: restore-parsimony rule ──────────────────────────────────
    has_affordable_primary = any(str(g.id) in affordable_set for g in primaries)
    if has_affordable_primary:
        candidate_pool = primaries
    else:
        # Promotion path: keep both.  Restore goals get scored by the
        # priority of the primary they un-block.
        candidate_pool = list(primaries) + list(restores)

    # ── Step 5: score ───────────────────────────────────────────────────
    selections: list[ResourceAwareSelection] = []
    for g in candidate_pool:
        gid = str(g.id)
        profile = profiles[gid]
        priority = float(getattr(g, "priority", 0.0))
        is_restore = _is_restore(profile)
        is_affordable_g = (gid in affordable_set)

        # Compute base score and rationale.
        if is_restore and not has_affordable_primary:
            if not primaries:
                # Exploration-mode promotion: no non-restore primary
                # is in the candidate pool at all (every primary was
                # filtered as unreachable / dependency-gated / not
                # declared this turn).  Score the restore by its OWN
                # priority — visiting the refuel is the productive
                # default when nothing else is in play.  See
                # ``SPEC_resource_aware_selector.md`` § "Restore-only
                # candidate set".
                if not is_affordable_g:
                    continue
                base_score = priority
                rationale = (f"no primaries reachable; restore chosen "
                             f"by own priority (priority={priority:.2f})")
            else:
                # Existing restore-promotion: primaries exist but none
                # affordable.  Score by priority of best un-blocked.
                best_p, best_id = _best_unblocked_priority(
                    profile, primaries, profiles, rlu, affordable_set,
                )
                if best_p <= 0.0:
                    continue  # this restore un-blocks nothing
                # Refuel-as-late principle: don't burn a one-shot
                # restore on a low-priority primary.  When the best
                # un-blocked primary is below the caller-supplied
                # priority floor, skip this restore — speculative
                # exploration / portal-discovery / VLM-hint goals at
                # priority 0.30-0.70 don't justify consuming a finite
                # resource.  Adapter passes ``0.0`` to keep the
                # legacy behaviour or a higher floor (e.g. 0.80) to
                # require task-level priority for promotion.
                if best_p < float(restore_promotion_priority_floor):
                    continue
                base_score = best_p
                rationale = (f"restore-promotion: un-blocks primary "
                             f"{best_id!r} (priority={best_p:.2f})")
        else:
            # Primary path.  Unaffordable primaries get score 0 and
            # are skipped (no point selecting an unaffordable goal —
            # the agent would just fail).
            if not is_affordable_g:
                continue
            base_score = priority
            rationale = (f"primary affordable "
                         f"(priority={priority:.2f})")

        # Horizon-N feasibility penalty.
        if horizon >= 2:
            simulated = _simulate_resource_state(profile, rlu)
            next_p = _next_most_important_primary(primaries, gid)
            if next_p is not None:
                next_pid = str(next_p.id)
                next_profile = profiles.get(next_pid)
                if next_profile is not None:
                    next_priority = float(getattr(next_p, "priority", 0.0))
                    still_afford = True
                    for k, cost in next_profile.cost.items():
                        if cost > simulated.get(k, 0):
                            still_afford = False
                            break
                    if not still_afford:
                        deduction = next_priority * float(blocking_penalty)
                        base_score -= deduction
                        rationale += (f"; horizon-2 penalty {deduction:.2f} "
                                      f"(would block {next_pid!r})")

        selections.append(ResourceAwareSelection(
            goal_id   = gid,
            rationale = rationale,
            profile   = profile,
            score     = base_score,
        ))

    if not selections:
        return None
    selections.sort(key=lambda s: s.score, reverse=True)
    best = selections[0]
    if best.score <= 0.0:
        return None
    return best
