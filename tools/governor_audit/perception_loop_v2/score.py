"""Per-cell agreement scoring against truth.json.

truth.json is the operator-verified answer key (produced offline
by relabel_lc1_truth.py).  It is NEVER read at runtime by the
classifier / detector / aggregator — only by this scoring module.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TurnScore:
    turn: int
    n_total: int
    n_correct: int
    # Per-code precision / recall.
    per_code: dict[str, dict[str, int]]
    # Cell-level diffs for inspection.
    diffs: list[tuple[int, int, str, str]]  # (row, col, predicted, truth)

    @property
    def agreement(self) -> float:
        return self.n_correct / max(1, self.n_total)


def score_turn(
    predicted: list[list[str]],
    truth: list[list[str]],
) -> TurnScore:
    rows = len(predicted)
    cols = len(predicted[0]) if rows else 0
    n_total = rows * cols
    n_correct = 0
    diffs: list[tuple[int, int, str, str]] = []
    per_code: dict[str, dict[str, int]] = {}
    all_codes: set[str] = set()
    for r in range(rows):
        for c in range(cols):
            p = predicted[r][c]
            t = truth[r][c]
            all_codes.add(p)
            all_codes.add(t)
    for code in all_codes:
        per_code[code] = {"tp": 0, "fp": 0, "fn": 0}
    for r in range(rows):
        for c in range(cols):
            p = predicted[r][c]
            t = truth[r][c]
            if p == t:
                n_correct += 1
                per_code[p]["tp"] += 1
            else:
                diffs.append((r, c, p, t))
                per_code[p]["fp"] += 1
                per_code[t]["fn"] += 1
    return TurnScore(
        turn=0,  # caller fills in
        n_total=n_total, n_correct=n_correct,
        per_code=per_code, diffs=diffs,
    )


@dataclass
class FixtureScore:
    """Aggregate score across a fixture's turns."""
    per_turn: list[TurnScore]
    n_total: int
    n_correct: int
    per_code: dict[str, dict[str, int]]

    @property
    def agreement(self) -> float:
        return self.n_correct / max(1, self.n_total)


def aggregate_scores(per_turn: list[TurnScore]) -> FixtureScore:
    n_total = sum(s.n_total for s in per_turn)
    n_correct = sum(s.n_correct for s in per_turn)
    per_code: dict[str, dict[str, int]] = {}
    for s in per_turn:
        for code, stats in s.per_code.items():
            if code not in per_code:
                per_code[code] = {"tp": 0, "fp": 0, "fn": 0}
            for k, v in stats.items():
                per_code[code][k] += v
    return FixtureScore(
        per_turn=per_turn,
        n_total=n_total, n_correct=n_correct, per_code=per_code,
    )
