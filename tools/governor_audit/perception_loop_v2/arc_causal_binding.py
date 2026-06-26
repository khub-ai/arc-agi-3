"""ARC binding for the causal-attribution loop — the game-specific glue.

The loop (causal_attribution.py) and the provider layer (causal_providers.py)
are domain-clean. This module supplies the ARC-offline instantiation of the
parts that ARE domain-specific:

  - `EnvCopySandbox`     — the offline ARC counterfactual sandbox: snapshot by
                           deepcopy of the env, restore by replacing it, apply
                           by adapter.step. (Online ARC would use an UndoSandbox
                           with ACTION7; a robot a sim rollback or undo.)
  - `swept_path_relations` — the general path projection: which structures lie
                           in the swept path of an action, so a culprit that is
                           not yet in contact still becomes a suspect (the
                           turn-99 lesson). Geometry only; given an action ->
                           direction map (learned from dynamics memory).
  - `make_relation`      — adapt relational_kinematics records to the loop type.

The remaining domain piece — `observe()` turning a frame into typed
relations/conditions — is the perception slot. In ARC it is the visual provider
(relational_kinematics over perceived bboxes + the goal-gap done-set); for the
counterfactual to read a *probe* frame it needs a deterministic per-frame
read, which is the per-level entity analysis / a CV segmenter. That dependency
(perception must segment standing structures, e.g. the shaft) is the same #1
the turn-99 review named; this module exposes the seam, it does not assume it.
"""
from __future__ import annotations

import copy
from typing import Callable, Optional

from causal_attribution import Relation
from causal_providers import SandboxProvider, Observation


# ---------------------------------------------------------------------------
# Offline ARC sandbox (CopySandbox instantiation via env deepcopy)
# ---------------------------------------------------------------------------

class EnvCopySandbox:
    """Counterfactual sandbox for the OFFLINE ARC env: deepcopy snapshot /
    replace restore / adapter.step apply. ~12 ms per snapshot (measured), so a
    full attribution (a few suspects x directions) is well under a second.

    Online ARC has no state copy -> use causal_providers.UndoSandbox with
    ACTION7 instead; the loop is identical."""

    # deepcopy of the ACTUAL game env = the real system, so a two-arm test here
    # earns 'confirmed' (not 'twin_confirmed'). A digital-twin sandbox would set
    # kind='twin' + a fault-class fidelity.
    kind = "real"
    fidelity = None

    def __init__(self, adapter):
        self._adapter = adapter

    def snapshot(self):
        return copy.deepcopy(self._adapter._env)

    def restore(self, snap) -> None:
        # deepcopy on restore so the same snapshot can be reused across probes
        self._adapter._env = copy.deepcopy(snap)

    def apply(self, action: str) -> None:
        self._adapter.step(action)


# ---------------------------------------------------------------------------
# Swept-path projection (the general "culprit not yet in contact" source)
# ---------------------------------------------------------------------------

def _bbox_sweep(bb, dr: int, dc: int, reach: int):
    """Grow a bbox (r1,c1,r2,c2) along (dr,dc) by `reach` cells — the region the
    entity would pass through under that motion."""
    r1, c1, r2, c2 = bb
    return (min(r1, r1 + dr * reach), min(c1, c1 + dc * reach),
            max(r2, r2 + dr * reach), max(c2, c2 + dc * reach))


def _overlap(a, b) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def swept_path_relations(bboxes: dict, movers: frozenset, direction,
                          structures: frozenset, reach: int = 8,
                          agent_name: str = "agent") -> list:
    """Relations naming a structure that lies in the swept PATH of the movers
    under `direction` (dr,dc) — i.e. a thing the carried set would traverse if
    the action were taken. This is what surfaces a culprit (e.g. a shaft) that
    is not yet touching the agent at the anomaly turn.

    Game-agnostic: pure geometry. `direction` is the action's learned effect
    vector (from dynamics memory); `structures` are the candidate fixed entities
    (walls/rails/shafts) to test against; `movers` are agent + coupled set."""
    if direction is None:
        return []
    dr, dc = direction
    out = []
    seen = set()
    for m in movers:
        if m not in bboxes:
            continue
        swept = _bbox_sweep(bboxes[m], dr, dc, reach)
        for s in structures:
            if s in movers or s not in bboxes or s in seen:
                continue
            if _overlap(swept, bboxes[s]):
                out.append(Relation("in_swept_path", (agent_name, s)))
                seen.add(s)
    return out


# ---------------------------------------------------------------------------
# Relation adapter + provider builder
# ---------------------------------------------------------------------------

def make_relation(record) -> Relation:
    kind = getattr(record, "kind", None) or record["kind"]
    ents = getattr(record, "entities", None) or record["entities"]
    return Relation(kind=kind, entities=tuple(ents))


# ---------------------------------------------------------------------------
# sk48 perception slot (deterministic CV) — OFFLINE TEST FIXTURE ONLY.
#
# WARNING — APPEARANCE-KEYED, NOT COMPETITION-SAFE. The helpers below match
# sk48's specific palette colors and the shaft's gray shade. They FAIL the
# adversarial regraphing test (recolor/retexture/resize the structures and they
# break), so they must NOT be used in the competition pipeline. They exist only
# to drive the deterministic offline counterfactual without a VLM call per step.
#
# The COMPETITION-SAFE structure source is `structure_detection.classify_
# structures(world)` — keyed on BEHAVIOR (static while others move) + role, never
# on color/pattern/size/orientation. In competition, perception is the VLM
# (appearance-agnostic), which segments the structure as a tracked entity, and
# the classifier confirms it by persistence. (Offline, an appearance-agnostic
# segmenter — background-subtraction CCA — is the remaining piece to retire the
# color-keyed fixture below.)
# ---------------------------------------------------------------------------

_SK48_COLORS = {
    "block_red": ((150, 0, 0), (255, 90, 90)),
    "block_orange": ((215, 105, 0), (255, 195, 85)),
    "block_blue": ((15, 85, 195), (95, 205, 255)),
    "block_green": ((35, 145, 35), (125, 235, 125)),
    "manipulator_tip": ((195, 0, 145), (255, 125, 255)),
}
_HUD_ROW0 = 53          # HUD strip starts here (64-row frame); play area above
_IDENT = {"block_red": "red", "block_orange": "orange",
          "block_blue": "blue", "block_green": "green"}


def _rgb(adapter):
    import numpy as np
    from game_adapter import _DEFAULT_PALETTE
    grid = adapter._normalise_frame(adapter._obs.frame)
    return np.array(_DEFAULT_PALETTE, dtype="uint8")[grid.astype(int) % 16]


def _mask_bbox(region, lo, hi):
    import numpy as np
    m = ((region[:, :, 0] >= lo[0]) & (region[:, :, 0] <= hi[0])
         & (region[:, :, 1] >= lo[1]) & (region[:, :, 1] <= hi[1])
         & (region[:, :, 2] >= lo[2]) & (region[:, :, 2] <= hi[2]))
    ys, xs = np.where(m)
    if not len(ys):
        return None
    return (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))


def _detect_shaft(play, block_bboxes, min_total=6, left_margin=13):
    """The shaft is a THIN, TALL vertical gray channel. Detect it by gray-pixel
    TOTAL per column (not contiguous run, because the blocks that sit ON the
    channel interrupt it): a channel column accumulates many gray pixels down
    its length, while the horizontal arm contributes only ~1-3 per column. The
    left rail is excluded by left_margin. Returns a tight bbox or None. (sk48
    instantiation; a robot would get structures from its map.)"""
    import numpy as np
    gray = ((abs(play[:, :, 0].astype(int) - play[:, :, 1]) < 25)
            & (abs(play[:, :, 1].astype(int) - play[:, :, 2]) < 25)
            & (play[:, :, 0] > 140) & (play[:, :, 0] < 220))
    h, w = gray.shape
    col_tot = gray.sum(axis=0)
    shaft_cols = [c for c in range(left_margin, w) if col_tot[c] >= min_total]
    if not shaft_cols:
        return None
    ys = np.where(gray[:, shaft_cols].any(axis=1))[0]
    return (int(ys.min()), min(shaft_cols), int(ys.max()), max(shaft_cols))


def _hud_done(rgb):
    """Completed targets = HUD swatches with a white border."""
    hud = rgb[_HUD_ROW0:]
    done = []
    for name, (lo, hi) in _SK48_COLORS.items():
        if name not in _IDENT:
            continue
        b = _mask_bbox(hud, lo, hi)
        if not b:
            continue
        y0, x0, y1, x1 = b
        import numpy as np
        ring = hud[max(0, y0 - 1):y1 + 2, max(0, x0 - 1):x1 + 2]
        if ((ring[:, :, 0] > 235) & (ring[:, :, 1] > 235)
                & (ring[:, :, 2] > 235)).sum() >= 3:
            done.append(_IDENT[name])
    return frozenset(done)


def _scene(rgb):
    """Return (bboxes, agents, structures) for the play area."""
    play = rgb[:_HUD_ROW0]
    bboxes = {}
    for name, (lo, hi) in _SK48_COLORS.items():
        b = _mask_bbox(play, lo, hi)
        if b:
            bboxes[name] = b
    blocks = {k: v for k, v in bboxes.items() if k.startswith("block_")}
    shaft = _detect_shaft(play, blocks)
    if shaft:
        bboxes["shaft"] = shaft
    return bboxes, blocks, ("shaft",) if shaft else ()


def sk48_observe_factory(adapter):
    """Build the `observe` callable for sk48 over the adapter's CURRENT frame.
    agents = the manipulator + the currently-threaded (carried) blocks."""
    def observe() -> Observation:
        rgb = _rgb(adapter)
        bboxes, blocks, _structs = _scene(rgb)
        done = _hud_done(rgb)
        agents = {"manipulator_tip"} | {f"block_{i}" for i in done
                                        if f"block_{i}" in bboxes}
        fp = tuple(sorted((k, tuple(v)) for k, v in bboxes.items())) + (tuple(sorted(done)),)
        return Observation(conditions=done, relations=(),
                           agents=frozenset(agents), fingerprint=fp)
    return observe


def learn_action_directions(adapter, agent_color="manipulator_tip") -> dict:
    """Discover each action's effect DIRECTION (dr,dc) by probing it on a
    deepcopy and measuring the agent's displacement — the dynamics-memory the
    swept-path projection needs, learned not injected."""
    sb = EnvCopySandbox(adapter)
    lo, hi = _SK48_COLORS[agent_color]
    dirs = {}
    for a in adapter.available_actions():
        if a in ("NONE", "CLICK", "ACTION7"):
            continue
        snap = sb.snapshot()
        before = _mask_bbox(_rgb(adapter)[:_HUD_ROW0], lo, hi)
        try:
            sb.apply(a)
            after = _mask_bbox(_rgb(adapter)[:_HUD_ROW0], lo, hi)
        finally:
            sb.restore(snap)
        if before and after:
            dr = (after[0] - before[0])
            dc = (after[1] - before[1])
            sgn = lambda x: (x > 0) - (x < 0)
            if dr or dc:
                dirs[a] = (sgn(dr), sgn(dc))
    return dirs


def _centroid(bb):
    return ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)


def sk48_project_path_factory(adapter):
    """Build the swept-path projector for sk48. A retract/pull drags the carried
    set TOWARD the anchor (the manipulator), so the swept path is the corridor
    from each carried block toward the agent — whatever structure lies between
    them (the shaft) is in the path. This needs no action-direction probing
    (which the anomaly action's own un-skewer would confound) and generalizes to
    'a pull drags the held set through what's between it and the anchor'.

    The shaft is a STANDING structure, captured ONCE here and held fixed across
    probes — re-detecting it each frame wrongly grows it downward with the
    moving rod stub, so it could never be cleared. (A robot would read fixed
    structures from its map; here we freeze the level's shaft bbox.)"""
    _b0, _, _ = _scene(_rgb(adapter))
    fixed_shaft = _b0.get("shaft")

    def project_path(action) -> list:
        rgb = _rgb(adapter)
        bboxes, blocks, _structs = _scene(rgb)
        done = _hud_done(rgb)
        agent = bboxes.get("manipulator_tip")
        if agent is None or fixed_shaft is None:
            return []
        bboxes = dict(bboxes)
        bboxes["shaft"] = fixed_shaft          # held fixed, not re-detected
        structs = ("shaft",)
        ar, ac = _centroid(agent)
        movers = [f"block_{i}" for i in done if f"block_{i}" in bboxes]
        out, seen = [], set()
        for m in movers:
            mb = bboxes[m]
            mr, mc = _centroid(mb)
            # corridor (bbox union) from the block to the agent
            corridor = (min(mb[0], int(ar)), min(mb[1], int(ac)),
                        max(mb[2], int(ar)), max(mb[3], int(ac)))
            for s in structs:
                if s in seen:
                    continue
                if _overlap(corridor, bboxes[s]):
                    out.append(Relation("in_swept_path",
                                        ("manipulator_tip", s)))
                    seen.add(s)
        return out
    return project_path


def build_arc_provider(adapter, observe: Callable[[], Observation],
                        project_path: Optional[Callable[[str], list]] = None
                        ) -> SandboxProvider:
    """Assemble a live ARC provider: caller-supplied `observe` (the perception
    slot) + the offline deepcopy sandbox + an optional swept-path projector.
    `observe` must read the adapter's CURRENT frame each call (so it reflects a
    probe's effect after apply)."""
    return SandboxProvider(
        observe=observe,
        list_actions=adapter.available_actions,
        sandbox=EnvCopySandbox(adapter),
        project_path=project_path or (lambda _a: []),
    )
