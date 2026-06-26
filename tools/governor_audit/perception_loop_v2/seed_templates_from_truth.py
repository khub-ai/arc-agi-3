"""Operator tool: seed the per-game template store from fixture
truth.json + frame.png pairs.

This is a TEACHING tool, NOT a runtime component.  It runs offline,
reads the operator-verified ground truth, and emits substrate-
agnostic visual templates (quantised-RGB top-K signatures) for the
roles the operator has labelled.  The runtime engine never authors
templates from truth; only this script does, and only when invoked
by the operator.

In production:
  - Templates are LEARNED by the aggregator from behaviour
    corroboration during live trials.
  - This tool is the seeding path used before the aggregator has
    enough live evidence — equivalent to operators describing
    \"what a green sediment looks like\" to the system, via the
    substrate-agnostic vocabulary (visual signature), not via
    game-specific code.

For every fixture turn, the tool:
  1. Detects entities via entity_detector.
  2. For each non-background entity, looks at the truth code of
     its centroid cell.  The centroid is the entity's "address";
     the truth code at that address is the role the operator's
     labelling intended.  (Pixel-weighted majority across all the
     entity's cells used to be the rule, but it mis-attributes any
     foreground entity whose footprint extends into many adjacent
     background cells — e.g. a wide bottom-row strip whose centre
     is on the budget counter but whose pixels mostly cover B.)
  3. Maps the truth code back to the role via the knowledge file's
     truth_codes mapping.
  4. Records the entity's visual_signature as a template for that
     role.

Templates with the same role and similar signatures are merged
(corroborations += 1).  Output: knowledge/templates/<game_id>.json.

Usage:
    python -m perception_loop_v2.seed_templates_from_truth \\
        [--game bp35]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE_PARENT = HERE.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from perception_loop_v2.observation import (  # noqa: E402
    load_fixture_frame, build_frame_observation,
)
from perception_loop_v2.entity_detector import detect_entities  # noqa: E402
from perception_loop_v2.knowledge_base import load_game  # noqa: E402
from perception_loop_v2.template_store import (  # noqa: E402
    Template, GameTemplates, save_templates, signature_similarity,
)


FIXTURES_ROOT = HERE.parent / "perception_tests" / "fixtures"


# Similarity threshold for merging templates of the same role.
# Above this overlap, two signatures are treated as the same
# template and corroborations += 1.
MERGE_THRESHOLD = 0.8


def _truth_for_cell(truth_grid, r: int, c: int) -> str:
    if r < 0 or r >= len(truth_grid):
        return "?"
    row = truth_grid[r]
    if c < 0 or c >= len(row):
        return "?"
    return row[c]


def _seed_one_fixture(
    fixture_dir: Path, kb,
) -> dict[tuple[str, str], list[tuple]]:
    """Returns {(level, role) → [(signature, pixel_count), ...]} from this fixture."""
    seq = json.loads(
        (fixture_dir / "sequence.json").read_text(encoding="utf-8")
    )
    turn_info = {int(t["turn"]): t for t in seq["turns"]}
    lc = str(seq.get("lc", 0))
    rules, truth_codes, _bg_proj = kb.for_level(int(lc))
    # Reverse mapping: truth_code → role.
    code_to_role = {v: k for k, v in truth_codes.items()}

    collected: dict[tuple[str, str], list[tuple]] = defaultdict(list)

    for td in sorted(fixture_dir.glob("turn_*")):
        try:
            n = int(td.name.split("_")[-1])
        except ValueError:
            continue
        info = turn_info.get(n)
        truth_path = td / "truth.json"
        frame_path = td / "frame.png"
        if (info is None or not truth_path.exists()
                or not frame_path.exists()):
            continue
        rgb = load_fixture_frame(frame_path)
        obs = build_frame_observation(
            rgb, turn=n, rows=8, cols=8,
            agent_position=tuple(info["agent_position"]),
            prev_observation=None,
        )
        entities = detect_entities(obs)
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        truth_grid = truth["cell_types"]

        for e in entities:
            if e.is_background_primary or e.is_background_secondary:
                continue
            # The entity's role comes from its CENTROID cell's truth
            # code.  The centroid is the entity's address; using the
            # truth at that address aligns the seeder with the
            # resolver's centroid projection.  An older variant used
            # pixel-weighted dominant truth across all the entity's
            # cells, which silently mis-attributed any entity whose
            # footprint extended into many background cells to the
            # background role (and then dropped it).
            cr, cc = e.centroid_cell
            centroid_code = _truth_for_cell(truth_grid, cr, cc)
            role = code_to_role.get(centroid_code)
            if not role:
                continue
            # Skip roles already covered by universal matchers — these
            # don't need visual templates.
            if role in ("agent", "background_primary",
                         "background_secondary"):
                continue
            collected[(lc, role)].append(
                (e.visual_signature, e.n_pixels)
            )

    return collected


def _merge_into_store(
    collected: dict[tuple[str, str], list[tuple]],
    gt: GameTemplates,
    trial_id: str,
) -> int:
    """Merge collected (signature, pixel_count) tuples into
    GameTemplates.  Returns the number of NEW templates added
    (after merging similar ones).

    Each template records the OBSERVED pixel_count range of its
    source entities — used at match time to reject candidates
    whose size is far outside the typical range.
    """
    added = 0
    for (lc, role), sig_px_pairs in collected.items():
        # Cluster similar signatures within this (lc, role) group.
        clusters: list[Template] = []
        for sig, px in sig_px_pairs:
            matched = False
            for cl in clusters:
                if signature_similarity(sig, cl.signature) >= MERGE_THRESHOLD:
                    cl.corroborations += 1
                    cl.pixel_count_min = min(cl.pixel_count_min, px)
                    cl.pixel_count_max = max(cl.pixel_count_max, px)
                    matched = True
                    break
            if not matched:
                clusters.append(Template(
                    role=role,
                    signature=sig,
                    level=lc,
                    first_seen_trial=trial_id,
                    corroborations=1,
                    pixel_count_min=px,
                    pixel_count_max=px,
                ))
        # Merge each cluster into the store.
        for new_t in clusters:
            existing = None
            for t in gt.templates:
                if t.role != new_t.role:
                    continue
                if t.level != new_t.level:
                    continue
                if signature_similarity(t.signature, new_t.signature) >= MERGE_THRESHOLD:
                    existing = t
                    break
            if existing is not None:
                existing.corroborations += new_t.corroborations
                if existing.pixel_count_min == 0:
                    existing.pixel_count_min = new_t.pixel_count_min
                else:
                    existing.pixel_count_min = min(
                        existing.pixel_count_min, new_t.pixel_count_min,
                    )
                existing.pixel_count_max = max(
                    existing.pixel_count_max, new_t.pixel_count_max,
                )
            else:
                gt.templates.append(new_t)
                added += 1
    return added


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--game", type=str, default="bp35",
                   help="game id (no harness suffix; default bp35)")
    args = p.parse_args()
    game_id = args.game
    kb = load_game(game_id)
    gt = GameTemplates(game_id=game_id)

    total_added = 0
    fixture_names = [
        d.name for d in sorted(FIXTURES_ROOT.iterdir())
        if d.is_dir() and d.name.startswith(f"{game_id}_")
        and (d / "sequence.json").exists()
    ]
    print(f"Seeding templates for {game_id} from "
          f"{len(fixture_names)} fixture(s)")
    for name in fixture_names:
        collected = _seed_one_fixture(
            FIXTURES_ROOT / name, kb,
        )
        added = _merge_into_store(collected, gt, trial_id=name)
        total_added += added
        for (lc, role), sigs in collected.items():
            print(f"  {name:<22} lc={lc} {role:<22} "
                  f"signatures={len(sigs)}")

    out_path = save_templates(gt)
    print()
    print(f"Wrote {len(gt.templates)} template(s) "
          f"({total_added} new) to {out_path}")
    for t in gt.templates:
        print(f"  role={t.role:<22} lc={t.level} "
              f"corroborations={t.corroborations}")


if __name__ == "__main__":
    main()
