"""Tactic knowledge base — sub-win-path multi-step tactics with
abstract preconditions, transferable across levels and games.

Sits between two existing layers:

  * ``global_priors.py`` — single (action, effect) priors, no
    sequences, no preconditions beyond action support count.
  * ``solutions_kb.py`` — full level-winning action sequences,
    keyed by (game_id, level).  No sub-win granularity.

A Tactic is a *named multi-step procedure that closes a subgoal*.
Examples (DISCOVERED, not injected):

  - side-swipe: "displace one of several co-located entities
    without engaging it" — used at sk48 lc=1 t126-141 to peel
    block_green off the row-13 stack via selective arm extension
    + perpendicular agent motion.

  - undo-first stuck-recovery: "when an action sequence dead-ends,
    reverse the most recent forward primitives before trying
    novel actions" — observed across sk48 lc=0 and lc=1.

Each Tactic carries ABSTRACT preconditions in open vocabulary so
the match step is game-agnostic.  Concrete action sequences
remain attached as evidence — they ground the abstract steps
when the strategy actor needs an example.

Storage: a single JSON file (default ``.tmp/tactic_kb.json``)
that survives across sessions and games, like
``global_action_priors.json``.

GAME-AGNOSTIC: preconditions, intent, abstract_steps all use the
WorldKnowledge open vocabulary (entity roles, qualitative
state-class features).  No hardcoded game names, no per-game
overrides.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


try:
    from kb_paths import kb_path as _kb_path
except ImportError:  # imported as a package
    from perception_loop_v2.kb_paths import kb_path as _kb_path
# Unified KB root (see kb_paths.py + docs/SPEC_knowledge_base.md).
DEFAULT_TACTIC_KB_PATH = _kb_path("tactic_kb.json")


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class TacticExample:
    """One concrete grounding of a tactic — the actual action
    sequence that succeeded in a specific game/level/turn window."""
    game_id: str
    level: int
    turn_range: list[int]      # [first_turn, last_turn]
    actions: list[str]         # concrete action_id strings as executed
    rationale_summary: str     # one-line: what the chain achieved
    outcome_summary: str       # one-line: what changed in the world


@dataclass
class Tactic:
    """A reusable sub-win-path tactic.

    A Tactic is matched against the CURRENT world snapshot by its
    ``preconditions`` dict.  The match semantics are deliberately
    loose: each precondition key is one of a small closed vocabulary
    of game-agnostic predicate kinds (see PRECONDITION_KINDS below),
    and the matcher answers "does this predicate hold in the
    current world?".

    Abstract steps are open-vocabulary natural-language descriptions
    of the procedure.  The strategy actor (VLM or planner) reads
    them as a hypothesis-strength template, NOT a literal action
    list.  Examples ground the abstract steps in concrete past
    successes.
    """
    tactic_id: str
    name: str
    intent: str
    preconditions: dict        # see PRECONDITION_KINDS
    abstract_steps: list[str]  # natural-language procedure
    expected_outcome: str
    examples: list[TacticExample] = field(default_factory=list)
    n_successes: int = 0
    n_failures: int = 0
    credence: float = 0.6      # base credence for a learned tactic


# Closed vocabulary of precondition predicate KINDS.  When a Tactic
# precondition uses one of these keys, ``match_tactic`` knows how to
# evaluate it against a WorldKnowledge snapshot.  Adapters can
# extend this vocabulary without touching tactic_kb itself by
# adding their own predicate evaluators to a registry.
PRECONDITION_KINDS = {
    # Number of entities sharing the same row (or column) at the
    # same time.  Value is an integer; predicate holds if at least
    # that many co-located entities exist.
    "min_co_located_entities_same_row",
    "min_co_located_entities_same_col",
    # Whether the agent has an entity it interacts via a horizontal
    # or vertical "arm" / extension.  Detected heuristically by the
    # presence of an entity whose role contains "arm" / "extension"
    # / "rope" / "tether" / "tool" attached to the agent.
    "agent_has_extension_tool",
    # Whether the agent is positioned at a different row (or col)
    # from a named group of entities — pre-aligned for sweep.
    "agent_row_distinct_from_targets",
    "agent_col_distinct_from_targets",
    # A forbid-effect constraint is active.  Value: the effect name
    # to avoid (e.g. "engagement", "consumption").
    "active_constraint_forbids_effect",
    # Free-form predicate description; used as a documentation
    # hook.  The matcher accepts these and treats them as "soft" —
    # advisory text shown to the strategy actor.
    "soft_predicate",
}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def load(path: Path = DEFAULT_TACTIC_KB_PATH) -> list[Tactic]:
    """Load the tactic KB from disk.  Returns an empty list on
    first-ever run (no file yet)."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[Tactic] = []
    for rd in data.get("tactics", []):
        examples = [TacticExample(**ex)
                    for ex in rd.get("examples", [])]
        rd = {**rd, "examples": examples}
        out.append(Tactic(**rd))
    return out


def save(tactics: list[Tactic],
         path: Path = DEFAULT_TACTIC_KB_PATH) -> None:
    """Persist the tactic KB to disk.  Atomic via tempfile rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"tactics": [asdict(t) for t in tactics]}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _entities_in_row_or_col_groups(world) -> tuple[int, int]:
    """Count the MAX number of currently-present entities sharing the
    same approximate row band, and the max sharing the same column
    band.  Heuristic: bin centroids by cell_ticks.  Returns
    (max_same_row, max_same_col)."""
    gi = world.grid_inference
    if gi is None or gi.cell_ticks is None:
        return (0, 0)
    step = gi.cell_ticks
    rows: dict[int, int] = {}
    cols: dict[int, int] = {}
    for r in world.entities.values():
        if r.last_seen_turn != world.turn:
            continue
        if r.current_bbox is None:
            continue
        cy = (r.current_bbox[0] + r.current_bbox[2]) / 2.0
        cx = (r.current_bbox[1] + r.current_bbox[3]) / 2.0
        rb = int(cy // step)
        cb = int(cx // step)
        # Only count entities likely to be interactable targets.
        role = (r.current_role or "").lower()
        if role in {"hud", "scenery", "decoration", "background",
                    "agent", "unknown"}:
            continue
        rows[rb] = rows.get(rb, 0) + 1
        cols[cb] = cols.get(cb, 0) + 1
    return (max(rows.values(), default=0),
            max(cols.values(), default=0))


def _agent_has_extension_tool(world) -> bool:
    """Heuristic: does the world currently contain a 'extension /
    arm / rope / tool' entity that's spatially connected to the
    agent?  Open-vocabulary role match."""
    EXTENSION_TOKENS = {
        "arm", "extension", "rope", "tether", "tool",
        "skewer", "rod", "tongue", "tongue-tip",
    }
    for r in world.entities.values():
        if r.last_seen_turn != world.turn:
            continue
        role = (r.current_role or "").lower()
        name = (r.name or "").lower()
        haystack = f"{role} {name}"
        if any(tok in haystack for tok in EXTENSION_TOKENS):
            return True
    return False


def _evaluate_precondition(world, key: str, value) -> bool:
    """Apply one precondition predicate against the current world.
    Unknown kinds default to True (soft-match) so the matcher is
    forward-compatible with adapter-added predicates."""
    if key == "min_co_located_entities_same_row":
        row_max, _col_max = _entities_in_row_or_col_groups(world)
        return row_max >= int(value)
    if key == "min_co_located_entities_same_col":
        _row_max, col_max = _entities_in_row_or_col_groups(world)
        return col_max >= int(value)
    if key == "agent_has_extension_tool":
        return _agent_has_extension_tool(world) == bool(value)
    if key in {"agent_row_distinct_from_targets",
               "agent_col_distinct_from_targets"}:
        # Heuristic: agent is currently NOT in the modal row/col band
        # of interactable entities.
        agent = next(
            (r for r in world.entities.values()
             if r.current_role == "agent"
             and r.last_seen_turn == world.turn),
            None,
        )
        if agent is None or agent.current_cell is None:
            return False
        gi = world.grid_inference
        if gi is None or not gi.cell_ticks:
            return False
        # Pick the modal row band of non-agent non-inert entities.
        bands: dict[int, int] = {}
        idx_for_key = (0 if "row" in key else 1)
        for r in world.entities.values():
            if r.last_seen_turn != world.turn or r.current_cell is None:
                continue
            if r is agent:
                continue
            role = (r.current_role or "").lower()
            if role in {"hud", "scenery", "decoration",
                         "background", "unknown"}:
                continue
            b = r.current_cell[idx_for_key]
            bands[b] = bands.get(b, 0) + 1
        if not bands:
            return False
        modal_band = max(bands.items(), key=lambda kv: kv[1])[0]
        return agent.current_cell[idx_for_key] != modal_band
    if key == "active_constraint_forbids_effect":
        # The constraint is supplied by the caller via the world's
        # symbolic_snapshot's ``constraints`` field if present.
        # Without that, default to True (advisory).
        return True
    if key == "soft_predicate":
        return True
    # Unknown kind: treat as soft.
    return True


def match_tactics(world, all_tactics: list[Tactic]) -> list[Tactic]:
    """Return the subset of tactics whose preconditions ALL hold in
    the current world.  Order: by descending credence × n_successes."""
    matched: list[Tactic] = []
    for t in all_tactics:
        ok = True
        for k, v in t.preconditions.items():
            if not _evaluate_precondition(world, k, v):
                ok = False
                break
        if ok:
            matched.append(t)
    matched.sort(key=lambda t: -(t.credence * max(1, t.n_successes)))
    return matched


# ---------------------------------------------------------------------------
# Surface for strategy prompt
# ---------------------------------------------------------------------------


def format_tactic_surface(world,
                            all_tactics: Optional[list[Tactic]] = None
                            ) -> str:
    """Render matching tactics as a block for inclusion in the
    strategy prompt.  Empty surface when no tactics match — the
    strategy layer should fall through to its default reasoning."""
    if all_tactics is None:
        all_tactics = load()
    if not all_tactics:
        return ("  (tactic KB is empty — no learned multi-step "
                "tactics on file yet)")
    matched = match_tactics(world, all_tactics)
    if not matched:
        return ("  (no stored tactics' preconditions match the "
                "current state)")
    lines: list[str] = []
    lines.append(
        "  !! LEARNED TACTICS that match the current world state:"
    )
    for t in matched[:5]:
        lines.append("")
        lines.append(
            f"  TACTIC {t.tactic_id!r} — {t.name} "
            f"(credence={t.credence:.2f}, successes={t.n_successes})"
        )
        lines.append(f"    Intent: {t.intent}")
        lines.append(
            f"    Matching preconditions: "
            f"{', '.join(f'{k}={v}' for k, v in t.preconditions.items())}"
        )
        lines.append("    Abstract steps:")
        for i, step in enumerate(t.abstract_steps, 1):
            lines.append(f"      {i}. {step}")
        lines.append(f"    Expected outcome: {t.expected_outcome}")
        if t.examples:
            ex = t.examples[0]
            lines.append(
                f"    Past success (1 of {len(t.examples)}): "
                f"{ex.game_id} lc={ex.level} t{ex.turn_range[0]}-"
                f"{ex.turn_range[1]}, actions="
                f"{','.join(ex.actions)}.  "
                f"Outcome: {ex.outcome_summary}"
            )
    lines.append("")
    lines.append(
        "  USAGE: a matched tactic is a STRONG PRIOR — treat the "
        "abstract steps as a hypothesised plan template and adapt "
        "to the current concrete actions / mechanics.  Skip the "
        "backward-reasoning protocol's open-ended search when a "
        "tactic match is clean."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


def promote_chain_as_tactic(
    tactic: Tactic,
    path: Path = DEFAULT_TACTIC_KB_PATH,
) -> None:
    """Add a new tactic (or merge into an existing one with the same
    tactic_id) and persist."""
    existing = load(path)
    by_id = {t.tactic_id: t for t in existing}
    if tactic.tactic_id in by_id:
        cur = by_id[tactic.tactic_id]
        cur.examples.extend(tactic.examples)
        cur.n_successes += tactic.n_successes
        # Asymmetric credence bump — supports raise faster than they fall.
        cur.credence = min(1.0, cur.credence + 0.05 * tactic.n_successes)
    else:
        existing.append(tactic)
    save(existing, path)
