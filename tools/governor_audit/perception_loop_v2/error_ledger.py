"""error_ledger.py -- the system collects its OWN significant errors, sets them aside, and learns.

When mistakes RECUR in a specific area -- telling the MOVER from the GOAL, missing a detail on a
LEGEND button, reading a gridded frame as set bars -- that area needs EXTRA scrutiny next time,
ESPECIALLY when a VARIATION (size, orientation, colour, position) has appeared, since a variation is
exactly what triggers the repeat. The ledger records errors by category; recurring categories raise a
heightened-scrutiny directive surfaced to the VLM.

This is meta-learning: don't just fix one mistake, REMEMBER the KIND of mistake, the SOLUTION that
resolved it, and guard against it -- and when a SIMILAR situation recurs, RECALL that experience.

Durable home: the ledger is a first-class KB store -- it lives under the unified KB root (``kb_path``)
alongside lessons / operators / solutions, and is registered in ``kb_paths.KNOWN_STORES`` so it
migrates + snapshots with the rest of the KB.  The closed learning loop (register -> index -> recall on
a similar situation -> reuse-or-formulate a solution -> register the solution) is orchestrated in
``learning_loop.py``, which links each error to the SOLUTION that resolved it.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

_STOP = set("the a an of to in on at is are be it its this that with for and or not no you your we "
            "as by from into out up down so if then a an one two when where what which how".split())


def _terms(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9_]+", (text or "").lower())
            if len(t) > 2 and t not in _STOP}


def _default_path():
    """The ledger's durable home under the unified KB root (falls back gracefully)."""
    try:
        from kb_paths import kb_path
        return kb_path("error_ledger.json")
    except Exception:
        return None


@dataclass
class ErrorRecord:
    category: str               # e.g. "mover_goal_identity", "legend_detail", "gridded_frame_read"
    description: str            # what went wrong, concretely
    variation: str = ""         # the variation that fooled it: size|orientation|colour|position|...
    fix: str = ""               # how it was (or should be) avoided (free-text guard)
    level: str = ""             # where it happened (game/level)
    resolution: str = ""        # the SOLUTION METHOD that actually resolved it (e.g. "playback-mined
                                # the box->direction map", "shape-match not orientation")
    solution_id: str = ""       # optional ref into solutions_kb (a replayable win-path) if one exists
    resolved: bool = False      # whether a working solution has been registered for this error

    def text(self) -> str:
        return " ".join([self.category, self.description, self.variation, self.fix, self.resolution])


class ErrorLedger:
    def __init__(self, path=None):
        # default to the unified KB root so errors are part of the durable KB
        self.path = Path(path) if path else _default_path()
        self.records: List[ErrorRecord] = []
        if self.path and self.path.exists():
            try:
                self.records = [ErrorRecord(**r) for r in json.loads(self.path.read_text(encoding="utf-8"))]
            except Exception:
                self.records = []

    # ---- registration -----------------------------------------------------
    def record(self, category, description, variation="", fix="", level="",
               resolution="", solution_id="") -> "ErrorRecord":
        # de-dup an identical (category, description) so re-runs don't inflate counts
        for r in self.records:
            if r.category == category and r.description == description:
                if variation and variation not in r.variation:
                    r.variation = (r.variation + "," + variation).strip(",")
                if fix and not r.fix:
                    r.fix = fix
                if resolution:
                    r.resolution, r.resolved = resolution, True
                if solution_id:
                    r.solution_id = solution_id
                self._save()
                return r
        rec = ErrorRecord(category, description, variation, fix, level, resolution, solution_id,
                          bool(resolution))
        self.records.append(rec)
        self._save()
        return rec

    def resolve(self, category, resolution, solution_id="", description="") -> int:
        """Attach the SOLUTION that fixed an area to its error record(s).  Closes the loop: a future
        recurrence in this area recalls the working solution instead of re-deriving.  Returns the
        number of records updated."""
        n = 0
        for r in self.records:
            if r.category == category and (not description or r.description == description):
                r.resolution, r.resolved = resolution, True
                if solution_id:
                    r.solution_id = solution_id
                n += 1
        if n:
            self._save()
        return n

    def _save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([asdict(r) for r in self.records], indent=2),
                                 encoding="utf-8")
        except Exception:
            pass

    # ---- indexing / aggregation ------------------------------------------
    def areas(self) -> dict:
        return dict(Counter(r.category for r in self.records))

    def recurring_areas(self, min_count: int = 2) -> dict:
        return {c: n for c, n in self.areas().items() if n >= min_count}

    def resolution_for(self, category: str) -> Optional[ErrorRecord]:
        """The most complete RESOLVED record for an area (the working solution to reuse), if any."""
        resolved = [r for r in self.records if r.category == category and r.resolved and r.resolution]
        if not resolved:
            return None
        return sorted(resolved, key=lambda r: (bool(r.solution_id), len(r.resolution)))[-1]

    # ---- situation-keyed recall ------------------------------------------
    def recall(self, situation, k: int = 4) -> List[ErrorRecord]:
        """Past errors RELEVANT to the current situation -- so a SIMILAR error recalls prior
        experience (and any solution that worked).  ``situation`` is free text or a term set; ranks by
        keyword overlap against each record, with a boost for resolved records and recurring areas."""
        q = situation if isinstance(situation, set) else _terms(str(situation))
        if not q:
            return []
        counts = self.areas()
        scored = []
        for r in self.records:
            overlap = len(q & _terms(r.text()))
            if overlap == 0:
                continue
            score = overlap + (1.0 if r.resolved else 0.0) + 0.3 * (counts.get(r.category, 1) - 1)
            scored.append((score, r))
        scored.sort(key=lambda t: -t[0])
        # de-dup by category, keep best
        seen, out = set(), []
        for _, r in scored:
            if r.category in seen:
                continue
            seen.add(r.category)
            out.append(r)
            if len(out) >= k:
                break
        return out

    def recall_surface(self, situation, k: int = 4) -> Optional[str]:
        """A situation-triggered recall directive: for each matching past error, surface the SOLUTION
        that worked (reuse it) or flag that none exists yet (formulate one).  This is the
        similar-error -> recall-experience step, ranked to NOW (vs the blanket scrutiny_note)."""
        hits = self.recall(situation, k=k)
        if not hits:
            return None
        lines = ["[RECALLED EXPERIENCE] this situation resembles areas where you have erred before "
                 "-- consult prior experience BEFORE deriving:"]
        for r in hits:
            if r.resolved and r.resolution:
                ref = f" (replayable solution {r.solution_id})" if r.solution_id else ""
                lines.append(f"  - {r.category}: SOLUTION that worked -> {r.resolution}{ref}. Reuse it.")
            else:
                guard = f" guard: {r.fix}" if r.fix else ""
                lines.append(f"  - {r.category}: no working solution recorded yet -- FORMULATE one "
                             f"(e.g. playback-mine prior levels / a discriminating probe).{guard}")
        return "\n".join(lines)

    def scrutiny_note(self, min_count: int = 2) -> Optional[str]:
        """A heightened-scrutiny directive for the error-prone areas, or None (blanket, situation-
        independent; complements the situation-keyed ``recall_surface``)."""
        rec = self.recurring_areas(min_count)
        if not rec:
            return None
        lines = []
        for cat, n in sorted(rec.items(), key=lambda kv: -kv[1]):
            vary = sorted({v for r in self.records if r.category == cat for v in r.variation.split(",") if v})
            fixes = sorted({r.fix for r in self.records if r.category == cat and r.fix})
            tail = (f"; variations that fooled it: {', '.join(vary)}" if vary else "")
            tail += (f"; guard: {fixes[0]}" if fixes else "")
            lines.append(f"  - {cat}: erred {n}x{tail}")
        return ("[ERROR-PRONE AREAS] You have repeatedly erred here -- invest EXTRA scrutiny and "
                "DOUBLE-CHECK against ground truth, ESPECIALLY when a variation (size/orientation/"
                "colour/position) appears, since a variation is what triggers the repeat:\n"
                + "\n".join(lines))
