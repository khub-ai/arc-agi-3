"""End-to-end driver: run the v2 perception loop on the bp35 lc=1
fixture and report per-turn + aggregate agreement against truth.json.

Per-turn pipeline:

  1. Read frame.png + agent_position.
  2. Build FrameObservation.
  3. Classify cells using the substrate-agnostic classifier.
  4. Score against truth.json.
  5. Detector proposes rule candidates from this turn's observation
     and classification.
  6. Aggregator ingests candidates; commits / promotes rules in the
     per-(game, lc) RuleStore.

At end of trial: aggregator.finalize_trial() reconciles cross-trial
gates.  RuleStore is saved.

Usage:
  python tools/governor_audit/perception_loop_v2/run.py
or
  python -m perception_loop_v2.run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow this file to be run as a script.
HERE = Path(__file__).resolve().parent
PACKAGE_PARENT = HERE.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from perception_loop_v2.observation import (
    load_fixture_frame, build_frame_observation, attach_persistence,
)
from perception_loop_v2.classifier import classify
from perception_loop_v2.detector import detect_for_turn
from perception_loop_v2.aggregator import Aggregator
from perception_loop_v2.rules import RuleStore
from perception_loop_v2.score import (
    score_turn, aggregate_scores, TurnScore,
)


FIXTURE_DIR = (
    HERE.parent / "perception_tests"
    / "fixtures" / "bp35_lc1_seq01"
)

RULES_ROOT = HERE.parent / "perception_loop_v2_rules"


def main() -> None:
    seq = json.loads(
        (FIXTURE_DIR / "sequence.json").read_text(encoding="utf-8")
    )
    turn_info = {int(t["turn"]): t for t in seq["turns"]}
    trial_id = "bp35_lc1_seq01"
    game_id = seq.get("game_id", "bp35")
    lc = int(seq.get("lc", 1))

    store = RuleStore.for_level(RULES_ROOT, game_id, lc)
    aggregator = Aggregator(store, trial_id=trial_id)

    observations = []
    per_turn_scores: list[TurnScore] = []
    per_turn_changed: list[tuple[int, list[str]]] = []

    for td in sorted(FIXTURE_DIR.glob("turn_*")):
        try:
            n = int(td.name.split("_")[-1])
        except ValueError:
            continue
        info = turn_info.get(n)
        if info is None:
            continue
        frame_path = td / "frame.png"
        truth_path = td / "truth.json"
        if not frame_path.exists() or not truth_path.exists():
            continue
        rgb = load_fixture_frame(frame_path)
        prev = observations[-1] if observations else None
        obs = build_frame_observation(
            rgb, turn=n, rows=8, cols=8,
            agent_position=tuple(info["agent_position"]),
            prev_observation=prev,
        )
        observations.append(obs)
        attach_persistence(observations, static_window=3)

        cls = classify(obs)

        # Score against operator-verified truth (scoring only — the
        # loop above did not read truth.json).
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        ts = score_turn(cls.cell_codes, truth["cell_types"])
        ts.turn = n
        per_turn_scores.append(ts)

        # Detector + aggregator pass: propose rule candidates and
        # commit / promote.
        candidates = detect_for_turn(obs, cls.cell_codes)
        changed = aggregator.ingest_turn(candidates, turn=n)
        if changed:
            per_turn_changed.append(
                (n, [
                    f"{r.status}:{r.type}:{r.body.get('role', '')}"
                    for r in changed
                ])
            )

    final_changed = aggregator.finalize_trial()
    store.save()

    # --- report ---
    agg = aggregate_scores(per_turn_scores)
    for ts in per_turn_scores:
        print(
            f"turn_{ts.turn:03d}: "
            f"{ts.n_correct}/{ts.n_total} = {ts.agreement:.3f}"
        )
    print()
    print(f"=== aggregate over {len(per_turn_scores)} turn(s):")
    print(f"  total cells = {agg.n_total}")
    print(f"  correct     = {agg.n_correct}")
    print(f"  agreement   = {agg.agreement:.4f}")
    print()
    print("  per-code precision / recall:")
    for code, st in sorted(agg.per_code.items()):
        tp = st["tp"]; fp = st["fp"]; fn = st["fn"]
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        print(f"    {code}: tp={tp:>4}  fp={fp:>3}  fn={fn:>3}  "
              f"prec={prec:.3f}  rec={rec:.3f}")

    print()
    print(f"=== rules learned (trial={trial_id}):")
    for r in store.all():
        body_summary = ""
        if r.type == "color_binding":
            body_summary = (
                f"role={r.body.get('role')} "
                f"rgb={tuple(r.body.get('rgb_key', ()))}"
            )
        elif r.type == "hud_strip":
            body_summary = f"y={tuple(r.body.get('y_range_logical', ()))}"
        elif r.type == "composite_sprite":
            cs = r.body.get("color_set", [])
            body_summary = (
                f"role={r.body.get('role')} colors={len(cs)} "
                f"bbox<={tuple(r.body.get('bbox_max_logical', ()))}"
            )
        print(
            f"  [{r.status}] {r.type:>16}  ev={r.evidence_count:>3}  "
            f"trials={len(r.supporting_trials)}  {body_summary}"
        )

    if per_turn_changed:
        print()
        print(f"=== rule-status changes by turn:")
        for n, events in per_turn_changed[:10]:
            print(f"  turn_{n:03d}: " + ", ".join(events))


if __name__ == "__main__":
    main()
