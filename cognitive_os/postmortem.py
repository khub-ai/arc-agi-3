"""PostMortem — episode retrospective, cross-episode learning gateway.

Called once at episode end (success, failure, timeout, or abandoned).
Its output is the mechanism by which the system accumulates knowledge
across episodes:

1. Lessons (structured :class:`StrategyClaim` / :class:`ConstraintClaim`
   hypotheses) get written back to the hypothesis store at
   :attr:`Scope.GAME` or broader, surviving the episode boundary.
2. Newly synthesised :class:`Option`\\s join the action-space cache.
3. Contradicted hypothesis signatures inform the Mediator at the
   next impasse to avoid repeating dead ends.
4. Mediator / Observer usage analytics feed budget recalibration.

Phase 4 scope
-------------
This module is the end-of-episode plumbing.  Heavy lifting like full
:class:`OptionSynthesiser` implementation (which requires pattern-
bearing Claim variants) is deferred to Phase 7 — a stub is included
here so the interface is stable and callers can expect the class to
exist even when it returns no options.

Capability audit
----------------
* **Tool creation** — PRIMARY for this module.  The
  :class:`OptionSynthesiser` hook is the designated place where
  recurring successful plan fragments get compressed into reusable
  Options.  Phase 4 wires the hook; Phase 7 populates it.
* **Debugging** — secondary.  Failed-plan analysis extracts
  :class:`StrategyClaim` adjustments that will reshape future
  OR-branch decisions in the planner.
* **Problem-solving** — secondary.  Lessons feed forward so the
  next episode starts with richer context.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Set

from .claims import (
    Claim,
    StrategyClaim,
)
from .conditions import AlwaysTrue, Condition
from . import hypothesis_store as _store
from .types import (
    Goal,
    GoalStatus,
    Hypothesis,
    Option,
    Plan,
    PlanStatus,
    PostMortem,
    Scope,
    ScopeKind,
    SurpriseEvent,
    WorldState,
)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run_post_mortem(ws:               WorldState,
                    episode_id:       str,
                    final_step:       int,
                    final_status:     str,
                    failed_plans:     Optional[List[Plan]] = None,
                    wall_time_seconds: float = 0.0,
                    *,
                    synthesise_options: bool = True) -> PostMortem:
    """Run the end-of-episode retrospective.

    Parameters
    ----------
    ws
        WorldState at episode end.  Used to extract surprises,
        contradictions, and final goal statuses.
    episode_id
        Stable identifier for this episode (typically
        ``f"{adapter.env_id}::ep{n}"``).
    final_step
        Step counter at termination.
    final_status
        Free-form reason: ``"success"``, ``"failure"``,
        ``"timeout"``, ``"abandoned"``, or a more specific adapter-
        defined string.
    failed_plans
        Plans that were invalidated or failed during the episode.
        The runner collects these as it goes and hands them off here.
    wall_time_seconds
        Real-time cost; populated by the runner.
    synthesise_options
        Whether to attempt Option synthesis from recurring
        successful-plan fragments.  Always safe to leave True;
        the synthesiser stub returns early if nothing qualifies.

    Returns
    -------
    PostMortem
        The retrospective record, including lessons to persist.

    Side effects
    ------------
    * Adds extracted :class:`StrategyClaim` lessons to the hypothesis
      store at ``Scope.GAME`` so they survive episode boundaries.
    * Registers newly synthesised :class:`Option`\\s in ``ws.options``
      so they are available to the planner in future episodes.
    """
    # Goal outcomes
    goal_outcomes = {g_id: goal.root.status
                     for g_id, goal in ws.goal_forest.goals.items()}

    # Collect surprise events from the observation history
    surprises: List[SurpriseEvent] = []
    for obs in ws.observation_history:
        for evt in obs.events:
            if isinstance(evt, SurpriseEvent):
                surprises.append(evt)

    # Hypotheses demoted during the episode — identified as those with
    # a non-null last_contradicted field and credence below commit.
    contradicted_ids = _extract_contradicted(ws)

    # Extract lessons: Phase 4 MVP extracts StrategyClaim adjustments
    # from goal outcomes.
    lessons = extract_lessons(ws, failed_plans or [], goal_outcomes,
                              final_status, final_step)

    # Persist lessons to the store at GAME scope so they survive
    # into future episodes.
    for lesson in lessons:
        _store.propose(
            ws,
            claim             = lesson,
            source            = "postmortem:lesson",
            scope             = Scope(kind=ScopeKind.GAME),
            step              = final_step,
            initial_credence  = 0.7,
            rationale         = "extracted from episode post-mortem",
        )

    # Option synthesis (stub in Phase 4; substantive in Phase 7)
    synthesised: List[str] = []
    if synthesise_options:
        synthesiser = OptionSynthesiser()
        synthesised = synthesiser.synthesise(ws, final_step)

    # Mediator / Observer usage — the runner updates these counters on
    # ws.agent under well-known keys; postmortem reads and clears.
    mediator_usage = dict(ws.agent.get("_mediator_usage") or {})
    observer_usage = dict(ws.agent.get("_observer_usage") or {})

    return PostMortem(
        episode_id               = episode_id,
        final_status             = final_status,
        final_step               = final_step,
        goal_outcomes            = goal_outcomes,
        failed_plans             = failed_plans or [],
        contradicted_hypotheses  = contradicted_ids,
        surprises                = surprises,
        lessons                  = lessons,
        options_synthesised      = synthesised,
        mediator_usage           = mediator_usage,
        observer_usage           = observer_usage,
        total_steps              = final_step,
        wall_time_seconds        = wall_time_seconds,
    )


# ---------------------------------------------------------------------------
# Lesson extraction
# ---------------------------------------------------------------------------


def extract_lessons(ws:           WorldState,
                    failed_plans: List[Plan],
                    goal_outcomes: Dict[str, GoalStatus],
                    final_status: str,
                    final_step:   int) -> List[Claim]:
    """Turn episode-level observations into persistable
    :class:`Claim`\\s.

    Phase 4 produces two kinds of lesson:

    * **Branch-outcome :class:`StrategyClaim`\\s** — for each OR-node
      that was active during a plan execution, record whether the
      chosen branch eventually led to the parent goal being
      achieved.  Success rate is crude (1.0 for success, 0.0 for
      failure, 0.5 for neutral/partial).  These claims update the
      planner's OR-branch bias in future episodes.

    * **Failure-context observations** — when ``final_status``
      indicates failure, emit a single ``StrategyClaim`` summarising
      the context under which the agent failed.  The
      ``context_pattern`` defaults to ``AlwaysTrue`` because Phase
      4 does not yet extract a structured context; Phase 7 will
      refine.

    Heavier lesson kinds (e.g. :class:`ConstraintClaim`\\s
    summarising repeated-failure corridors) require machinery that
    lives in later phases.
    """
    lessons: List[Claim] = []

    # Branch-outcome lessons from successful plans
    for plan in failed_plans:
        if plan.status != PlanStatus.COMPLETE:
            continue
        for or_node_id, chosen_child_id in plan.branch_selections.items():
            # Did the plan's goal eventually succeed?
            parent_status = goal_outcomes.get(plan.goal_id)
            success = parent_status == GoalStatus.ACHIEVED
            lessons.append(StrategyClaim(
                context_pattern = AlwaysTrue(),
                strategy_type   = chosen_child_id,
                success_rate    = 1.0 if success else 0.0,
                n_trials        = 1,
            ))

    # Global failure lesson
    if final_status.startswith("failure") or final_status == "timeout":
        lessons.append(StrategyClaim(
            context_pattern = AlwaysTrue(),
            strategy_type   = f"avoid:{final_status}",
            success_rate    = 0.0,
            n_trials        = 1,
        ))

    return lessons


# ---------------------------------------------------------------------------
# Option synthesis (stub)
# ---------------------------------------------------------------------------


class OptionSynthesiser:
    """Detect recurring successful plan fragments and promote them
    to :class:`Option`\\s.

    Phase 4 provides only the scaffolding; substantive synthesis
    arrives in Phase 7 when pattern-bearing Claim variants enable
    proper anti-unification across plan fragments.  The stub
    maintains the end-to-end pipeline so runners and tests can
    rely on the interface existing.

    What Phase 7 will add:

    * Anti-unification across successful plan fragments that share
      structure but differ in concrete parameters (positions,
      entity IDs, magnitudes).
    * Parameter extraction — the varying parts become
      :attr:`Option.parameters`.
    * Applicability inference — the invariant parts of the fragments'
      pre-conditions become :attr:`Option.applicability`.
    * Promotion — after N successful invocations across multiple
      games, :attr:`Option.scope` is upgraded from ``GAME`` to
      ``GLOBAL``.

    The current stub records that synthesis was attempted and logs
    eligible fragment groups on ``ws.agent["_option_candidates"]``
    so Phase 7's implementation can pick up from there.
    """

    def synthesise(self, ws: WorldState, step: int) -> List[str]:
        """Look for Option candidates and (in Phase 7) create them.

        Returns
        -------
        list[str]
            IDs of newly registered Options.  Empty in Phase 4 — the
            stub does not yet construct Options.
        """
        candidates = self.find_candidate_fragments(ws)
        # Record for diagnostics / Phase 7 consumption.
        ws.agent["_option_candidates"] = candidates
        # Phase 4 does not construct Options.
        return []

    def find_candidate_fragments(self, ws: WorldState) -> List[Dict]:
        """Scan ``ws.observation_history`` and completed plans for
        recurring successful fragments.

        Phase 4 returns a coarse diagnostic: groups of transition
        hypotheses sharing the same ``action`` that were used
        together in successful plans more than once.  This is a
        proxy for "there is a recurring useful pattern here" without
        fully constructing the pattern.

        Phase 7 will replace this with real anti-unification.
        """
        # Light-weight proxy: count how often each action appears in
        # committed TransitionClaims.  Actions used frequently and
        # successfully are strong Option candidates.
        from .claims import TransitionClaim  # local import to avoid cycle at module top
        action_counts: Counter = Counter()
        for h in _store.committed(ws):
            if isinstance(h.claim, TransitionClaim):
                action_counts[h.claim.action] += 1
        return [
            {"action": action, "transition_count": count}
            for action, count in action_counts.most_common()
            if count >= 3
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_contradicted(ws: WorldState) -> List[str]:
    """Return IDs of hypotheses that received contradicting evidence
    during the episode and are now either below commit threshold or
    already pruned (not in ws.hypotheses).  For active hypotheses,
    the check looks at last_contradicted being set."""
    out: List[str] = []
    for h_id, h in ws.hypotheses.items():
        if h.credence.last_contradicted is not None:
            # Only flag if credence is actually below commit now
            cfg = _store._credence_cfg(ws)
            if not h.credence.is_committed(cfg):
                out.append(h_id)
    return out
