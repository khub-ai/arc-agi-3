"""Game-agnostic PLANNING PRIORS store + loader.

WHY THIS EXISTS
---------------
General strategic knowledge ("when you cannot reach a target directly, act
through an intermediary"; "a solid that matches a congruent hole => the win is
to seat it"; "a mark is a relational signal, not noise") was authored as
structured priors in ``kb_seed_candidates/general_priors_planning.json`` but had
NO loader and NO consumer -- so it was inert.  This module is the loader: it
reads the cold-seeded ``planning_priors.json`` store and renders each prior into
a recall-surface line, so ``kb_recall`` can surface the RIGHT prior for the live
situation (the same channel as ``general_knowledge``).

The priors keep their STRUCTURE (``trigger`` / ``difference`` / ``operator`` /
``spawns_subgoals`` / ``reasoning_mode``) so a future means-ends consumer can
chain the backward operators into a plan; this module only does the
text-recall consumer (VLM-in-control), which is what makes them live today.

Schema (JSON): either a bare list of priors, or
``{"note": ..., "priors": [ {prior}, ... ]}`` where each prior is::

    {"id", "kind", "reasoning_mode": "forward"|"backward",
     "trigger"?: str,        # forward priors: the situation that fires them
     "difference"?: str,     # backward priors: the difference they reduce
     "operator": str,        # what to do
     "spawns_subgoals"?: [str],
     "credence": float, "provenance": str}

Game-agnostic by construction (Tier-0: no game ids / ACTION<n> / hex colours),
so the recall gate surfaces them on every game.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

try:
    from kb_paths import kb_path as _kb_path
except ImportError:                                  # imported as a package
    from perception_loop_v2.kb_paths import kb_path as _kb_path

DEFAULT_PLANNING_PATH = _kb_path("planning_priors.json")


def load_planning_priors(path: Path = DEFAULT_PLANNING_PATH) -> List[dict]:
    """All planning priors as a list of dicts (empty when the store is absent or
    unreadable -- a missing store must never crash recall)."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(blob, list):
        return blob
    return blob.get("priors") or []


def render_text(prior: dict) -> str:
    """Render one prior into a single recall-surface line: the SITUATION it fires
    in (its forward ``trigger`` or backward ``difference``) followed by the
    ``operator`` to apply.  Returns '' for a prior with no operator (skipped by
    the caller) so a malformed entry never surfaces as noise."""
    op = (prior.get("operator") or "").strip()
    if not op:
        return ""
    cond = (prior.get("trigger") or prior.get("difference") or "").strip()
    body = f"WHEN {cond}: {op}" if cond else op
    subs = prior.get("spawns_subgoals") or []
    if subs:
        body += " (subgoals: " + "; ".join(str(s) for s in subs) + ")"
    return body


def recall_kind(prior: dict) -> str:
    """Map a prior's reasoning mode to a ``kb_recall`` ranking kind.

    FORWARD priors (a trigger that hypothesizes a goal / win, e.g. figure-ground
    complement = delivery, or a mark = a correspondence/role) are DECLARATIVE
    'technique' -- surfaced even at level start so the win hypothesis front-loads.
    BACKWARD priors (means-ends OPERATORS that reduce a difference) are 'operator'
    -- an answer-kind that surfaces on keyword match and is boosted when stuck,
    which is exactly when a difference-reducing operator is needed."""
    return "technique" if (prior.get("reasoning_mode") == "forward") else "operator"


def as_pool_entries(path: Path = DEFAULT_PLANNING_PATH) -> List[dict]:
    """Planning priors rendered as ``kb_recall`` candidate-pool entries
    ({source, kind, text, credence, id}).  The single integration point for
    recall; guarded so it degrades to [] on any error."""
    out: List[dict] = []
    try:
        for pr in load_planning_priors(path):
            text = render_text(pr)
            if not text:
                continue
            out.append({
                "source": "planning_prior",
                "kind": recall_kind(pr),
                "text": text,
                "credence": float(pr.get("credence", 0.4)),
                "id": str(pr.get("id", "")),
            })
    except Exception:
        return []
    return out
