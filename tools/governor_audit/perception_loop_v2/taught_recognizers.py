"""VLM-taught substrate recognizers.

THE IDEA
--------
The VLM is good at recognizing some things visually (e.g. "this block is
impaled on the rod" — a gestalt) but it is EXPENSIVE to ask it every turn.
So the VLM recognizes it once (or occasionally), and TEACHES the substrate a
cheap, deterministic recognizer that runs every turn with no VLM call. The
VLM stays the arbiter: it spot-checks and re-teaches if the recognizer drifts.

This is the tool-offload principle made self-service and general: any
recurring recognition the VLM can do but shouldn't do every turn, it offloads
to a substrate recognizer it teaches. Game-agnostic — nothing here knows
"impaled" or "sk48"; the VLM supplies the label + the examples.

TEACHING FORM (simplest first): labeled EXEMPLARS. The VLM labels some tracked
instances ("#1,#2,#3 = impaled; #7 = free"); the substrate captures each
labeled instance's CONTEXT bitmap (the instance bbox expanded by a margin, so
the surrounding rod / empty space is included — that margin is what
distinguishes impaled from free). Each turn it classifies every instance by
nearest-exemplar match over the context bitmap. Confidence = best similarity
and the margin to the runner-up class; low-confidence instances are flagged
back to the VLM.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    from PIL import Image
    _OK = True
except Exception:
    _OK = False


def _load(path):
    try:
        return np.array(Image.open(path).convert("RGB"), dtype=np.int16)
    except Exception:
        return None


def _window(frame, bbox, margin: int):
    """A fixed window = the instance's bbox padded by `margin` on each side,
    with out-of-bounds cells set to a -1 sentinel. Returns (window int16
    HxWx3, (block_h, block_w)). NOT resized: same-type instances (same block
    size) yield same-shape windows with the block at the same offset
    [margin:margin+bh, margin:margin+bw], so the SURROUND aligns cell-for-cell.
    The surround is where the discriminating context (rod vs empty) lives."""
    t, l, b, r = bbox
    bh, bw = b - t + 1, r - l + 1
    H, W = frame.shape[:2]
    win = np.full((bh + 2 * margin, bw + 2 * margin, 3), -1, dtype=np.int16)
    r0, c0 = t - margin, l - margin
    rr0, cc0 = max(0, r0), max(0, c0)
    rr1 = min(H, t + bh + margin)
    cc1 = min(W, l + bw + margin)
    if rr1 > rr0 and cc1 > cc0:
        win[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = frame[rr0:rr1, cc0:cc1]
    return win, (bh, bw)


def _surround_similarity(a, b, bsize, margin: int, tol: int = 40) -> float:
    """Match fraction over the SURROUND ring only — the block interior is
    masked out (it's identical for all instances of a type, so it would just
    dilute the discriminating signal). Out-of-bounds sentinel cells excluded."""
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    bh, bw = bsize
    mask = np.ones(a.shape[:2], dtype=bool)
    mask[margin:margin + bh, margin:margin + bw] = False     # drop interior
    valid = mask & (a[:, :, 0] >= 0) & (b[:, :, 0] >= 0)     # drop OOB
    nval = int(valid.sum())
    if nval == 0:
        return 0.0
    diff = np.abs(a - b).sum(axis=2)
    return float(((diff <= tol * 3) & valid).sum() / nval)


@dataclass
class Exemplar:
    label: str
    patch: list                      # nested list (json-able) HxWx3 window
    bsize: Tuple[int, int] = (0, 0)  # (block_h, block_w) inside the window
    inst_id: str = ""
    turn: int = 0


@dataclass
class TaughtRecognizer:
    """A recognizer the VLM taught from labeled exemplars."""
    name: str                        # e.g. 'impaled'
    target_types: List[str] = field(default_factory=list)
    margin: int = 4
    patch_size: Tuple[int, int] = (16, 16)
    exemplars: List[Exemplar] = field(default_factory=list)
    min_confidence: float = 0.15     # gap to runner-up below which we abstain
    provenance: dict = field(default_factory=dict)

    def labels(self) -> List[str]:
        return sorted({e.label for e in self.exemplars})


# ---------------------------------------------------------------------------
# Teach (VLM authors) + classify (substrate runs cheaply)
# ---------------------------------------------------------------------------

def teach(name: str, target_types: List[str], frame, instances: dict,
          labels: Dict[str, str], *, margin: int = 4,
          patch_size: Tuple[int, int] = (16, 16),
          existing: Optional[TaughtRecognizer] = None) -> TaughtRecognizer:
    """Capture context exemplars for the VLM-labeled instances and add them to
    a recognizer (new or existing — teaching is incremental). `instances` is
    {inst_id: object-with-.bbox}; `labels` is {inst_id: label}."""
    if not _OK:
        return existing or TaughtRecognizer(name=name, target_types=list(target_types))
    arr = _load(frame) if not isinstance(frame, np.ndarray) else frame
    rec = existing or TaughtRecognizer(
        name=name, target_types=list(target_types), margin=margin,
        patch_size=tuple(patch_size),
        provenance={"taught_by": "vlm"})
    if arr is None:
        return rec
    for iid, label in labels.items():
        inst = instances.get(iid)
        if inst is None:
            continue
        win, bsize = _window(arr, inst.bbox, rec.margin)
        rec.exemplars.append(Exemplar(
            label=str(label), patch=win.astype(int).tolist(), bsize=tuple(bsize),
            inst_id=iid, turn=int(getattr(inst, "_turn", 0))))
    return rec


def classify_instance(rec: TaughtRecognizer, frame, inst) -> dict:
    """Cheap per-instance classification by nearest-exemplar match. Returns
    {label, confidence, margin_to_runner_up, abstain}."""
    if not _OK or not rec.exemplars:
        return {"label": None, "abstain": True, "reason": "no recognizer"}
    arr = _load(frame) if not isinstance(frame, np.ndarray) else frame
    if arr is None:
        return {"label": None, "abstain": True, "reason": "no frame"}
    win, bsize = _window(arr, inst.bbox, rec.margin)
    tol = 40
    # One representative window per label (comparable block size only).
    reps: Dict[str, "np.ndarray"] = {}
    for ex in rec.exemplars:
        if tuple(ex.bsize) != tuple(bsize):
            continue
        if ex.label not in reps:
            reps[ex.label] = np.array(ex.patch, dtype=np.int16)
    if not reps:
        return {"label": None, "abstain": True, "reason": "no comparable exemplar"}
    # DISCRIMINATIVE cells: where the label representatives DIFFER from each
    # other (e.g. rod-vs-empty). Matching only there isolates the signal that
    # actually distinguishes the classes, so a small-but-decisive cue (a rod
    # flanking the block) yields a confident verdict instead of being washed
    # out by the identical block interior + identical empty surround.
    lab_list = list(reps)
    bh, bw = bsize
    interior = np.zeros(win.shape[:2], dtype=bool)
    interior[rec.margin:rec.margin + bh, rec.margin:rec.margin + bw] = True
    disc = np.zeros(win.shape[:2], dtype=bool)
    for i in range(len(lab_list)):
        for j in range(i + 1, len(lab_list)):
            d = np.abs(reps[lab_list[i]] - reps[lab_list[j]]).sum(axis=2)
            disc |= (d > tol * 3)
    disc &= ~interior & (win[:, :, 0] >= 0)
    best_by_label: Dict[str, float] = {}
    for lab, rep in reps.items():
        m = disc & (rep[:, :, 0] >= 0)
        nm = int(m.sum())
        if nm == 0:
            # no discriminative cells vs others -> fall back to surround match
            best_by_label[lab] = _surround_similarity(win, rep, bsize, rec.margin)
            continue
        diff = np.abs(win - rep).sum(axis=2)
        best_by_label[lab] = float(((diff <= tol * 3) & m).sum() / nm)
    ranked = sorted(best_by_label.items(), key=lambda kv: -kv[1])
    top_label, top_sim = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0.0
    gap = top_sim - runner
    abstain = gap < rec.min_confidence
    return {"label": None if abstain else top_label, "confidence": round(top_sim, 3),
            "margin_to_runner_up": round(gap, 3), "abstain": abstain,
            "best_label": top_label}


def apply_all(rec: TaughtRecognizer, frame, instances: dict) -> dict:
    """Classify every instance of the recognizer's target types. Returns
    {inst_id: result}. The VLM reads the labels; abstentions are the cases it
    should re-recognize itself (and possibly re-teach)."""
    out = {}
    for iid, inst in instances.items():
        if rec.target_types and getattr(inst, "type_name", None) \
                not in rec.target_types:
            continue
        out[iid] = classify_instance(rec, frame, inst)
    return out


def format_recognizer_surface(rec: TaughtRecognizer, results: dict) -> str:
    """Render the taught-recognizer verdicts + abstentions for the VLM."""
    if not results:
        return ""
    counts: Dict[str, int] = {}
    abstain = []
    for iid, r in results.items():
        if r.get("abstain"):
            abstain.append(iid)
        else:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
    summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    lines = [f"TAUGHT RECOGNIZER '{rec.name}' (substrate-applied, no VLM call): "
             f"{summary or 'no confident verdicts'}"]
    for iid, r in sorted(results.items()):
        if r.get("abstain"):
            continue
        lines.append(f"    {iid}: {r['label']} (conf {r['confidence']})")
    if abstain:
        lines.append(f"    ABSTAINED (re-recognize yourself + re-teach): "
                     f"{', '.join(sorted(abstain))}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence (per game)
# ---------------------------------------------------------------------------

def _path(game_id: str) -> Path:
    try:
        from subroutine_kb import DEFAULT_SUBROUTINE_KB_PATH as _p
        return Path(_p).parent / f"recognizers_{game_id}.json"
    except Exception:
        return Path(f"recognizers_{game_id}.json")


def load_recognizers(game_id: str) -> Dict[str, TaughtRecognizer]:
    p = _path(game_id)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        out = {}
        for name, d in data.items():
            d["patch_size"] = tuple(d.get("patch_size", (16, 16)))
            d["exemplars"] = [Exemplar(**e) for e in d.get("exemplars", [])]
            out[name] = TaughtRecognizer(**d)
        return out
    except Exception:
        return {}


def save_recognizers(game_id: str, recs: Dict[str, TaughtRecognizer]) -> None:
    p = _path(game_id)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({n: asdict(r) for n, r in recs.items()}))
    except Exception:
        pass
