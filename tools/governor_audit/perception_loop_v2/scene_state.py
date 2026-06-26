"""scene_state.py -- the canonical, structured Scene State (state-as-medium).

ONE source of truth for the scene.  Communication (VLM<->substrate and COS<->user)
happens over THIS structured state via refinement ops + queries; the entity image
and the trace page become VIEWS rendered from it, not the medium.  This consolidates
the scattered entity geometry (foreground_components / structural_grid / decode_panel /
entity_grounding) + the standalone entity_resolve into one stateful layer.

Principles:
  * SUBSTRATE measures geometry (exact bboxes + click points + stable ids); the VLM
    assigns ROLES by id and drives refinement -- it never supplies pixel coords.
  * LAZY + incremental resolution: entities start COARSE; they are decomposed into
    children only on request (`resolve`), so the background checkerboard is never
    shattered into thousands of tiles unless asked.
  * The symbolic state is FALLIBLE.  An entity can be flagged needs_inspection so the
    VLM `inspect`s the rendered crop and recovers from mis-extraction -- the state is
    the efficient channel, the image is the ground-truth backstop.
  * Every op appends to an event log; snapshot() serialises the whole state for the
    live views + the archive.

Coords are tick (logical-frame) units; a click point is (col, row) for CLICK:col,row.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# -----------------------------------------------------------------------------
# Substrate geometry (measured, exact) -- the authority on WHERE things are.
# -----------------------------------------------------------------------------

def _structural_field(region):
    """Figure-ground by STRUCTURE: FIELD = largest connected uniform-colour
    component; foreground = the rest.  Never a modal-colour key.  Returns
    (fg_mask, field_rgb)."""
    import silhouette_track as _ST
    reg = np.ascontiguousarray(region).astype(int)
    if reg.size == 0:
        return np.zeros(reg.shape[:2], dtype=bool), (0, 0, 0)
    packed = (reg[:, :, 0] << 16) | (reg[:, :, 1] << 8) | reg[:, :, 2]
    field_c, fcol = None, 0
    for cv in (int(v) for v in np.unique(packed)):
        for comp in _ST.connected_components(packed == cv):
            if field_c is None or comp[6] > field_c[6]:
                field_c, fcol = comp, cv
    fg = np.ones(packed.shape, dtype=bool)
    if field_c is not None:
        fg[field_c[0]:field_c[2] + 1, field_c[1]:field_c[3] + 1] &= ~np.asarray(field_c[7], dtype=bool)
    return fg, (fcol >> 16 & 255, fcol >> 8 & 255, fcol & 255)


def measure_components(frame_rgb, bbox) -> list:
    """Every foreground connected component in bbox: {bbox, click:[col,row], h, w,
    area, orient}.  Edge-to-edge borders dropped (structural).  Sorted row-major.

    bbox is [r0, c0, r1, c1] -- ROW-first (top, left, bottom, right), NOT (x,y).  The
    per-component bbox is also [r0,c0,r1,c1]; only `click` is [col,row] (the engine's
    CLICK order).  Passing an (x,y)-ordered bbox silently reads the wrong region (e.g.
    clipping columns) -- a recurring footgun; keep row-first here."""
    import silhouette_track as _ST
    try:
        r0, c0, r1, c1 = [int(v) for v in bbox]
        reg = np.asarray(frame_rgb)[r0:r1 + 1, c0:c1 + 1, :3]
        hreg, wreg = reg.shape[:2]
        fg, _f = _structural_field(reg)
        out = []
        for cm in _ST.connected_components(fg):
            br0, bc0, br1, bc1, area = cm[0], cm[1], cm[2], cm[3], cm[6]
            if area < 1:
                continue
            h, w = br1 - br0 + 1, bc1 - bc0 + 1
            if (bc0 <= 1 and bc1 >= wreg - 2) or (br0 <= 1 and br1 >= hreg - 2):
                continue                                   # border/separator
            out.append({"bbox": [r0 + br0, c0 + bc0, r0 + br1, c0 + bc1],
                        "click": [int(c0 + (bc0 + bc1) // 2), int(r0 + (br0 + br1) // 2)],
                        "h": int(h), "w": int(w), "area": int(area),
                        "orient": "h" if w > h else ("v" if h > w else "sq")})
        out.sort(key=lambda e: (e["bbox"][0], e["bbox"][1]))
        return out
    except Exception:
        return []


def _cluster(vals, tol=2):
    if not vals:
        return []
    vals = sorted(vals)
    groups = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [sum(g) / len(g) for g in groups]


def _pitch(centers) -> Optional[float]:
    """Median spacing between consecutive cluster centres (the grid pitch).  None
    if fewer than two."""
    if not centers or len(centers) < 2:
        return None
    diffs = sorted(centers[i + 1] - centers[i] for i in range(len(centers) - 1))
    return diffs[len(diffs) // 2]


def measure_grid(frame_rgb, bbox, tol=2) -> Optional[dict]:
    """Resolve a panel/grid into bars with (row,col) positions by clustering the
    measured component centres.  Returns {n_rows, n_cols, bars:[{rc:(r,c), click,
    bbox, orient}]} or None.

    Robust to BORDERS/SEPARATORS -- including a gapped full-width border that the
    edge-to-edge filter misses (the gap splits it into wide pieces).  A genuine cell
    mark FITS WITHIN ONE GRID PITCH; a separator spans MULTIPLE cells.  So pass 1
    derives the pitch from all components, pass 2 keeps only components no larger
    than the pitch, then re-clusters.  Pitch is derived from the grid, not tuned."""
    def cy(e):
        return (e["bbox"][0] + e["bbox"][2]) / 2

    def cx(e):
        return (e["bbox"][1] + e["bbox"][3]) / 2

    comps = measure_components(frame_rgb, bbox)
    if len(comps) < 2:
        return None
    rp = _pitch(_cluster([cy(e) for e in comps], tol))
    cp = _pitch(_cluster([cx(e) for e in comps], tol))
    if rp and cp:
        comps = [e for e in comps if e["w"] <= cp and e["h"] <= rp]
    if len(comps) < 2:
        return None
    rc = _cluster([cy(e) for e in comps], tol)
    cc = _cluster([cx(e) for e in comps], tol)
    if not rc or not cc:
        return None

    def nearest(v, cs):
        return min(range(len(cs)), key=lambda i: abs(cs[i] - v))
    bars = []
    for e in comps:
        r = nearest(cy(e), rc)
        c = nearest(cx(e), cc)
        bars.append({"rc": (r, c), "click": e["click"], "bbox": e["bbox"], "orient": e["orient"]})
    return {"n_rows": len(rc), "n_cols": len(cc), "bars": bars}


# -----------------------------------------------------------------------------
# The Scene State (stateful layer over the measured geometry)
# -----------------------------------------------------------------------------

@dataclass
class Entity:
    id: str
    bbox: list                          # [r0,c0,r1,c1] tick
    click: list                         # [col,row] measured centre (click-by-id)
    kind: str = "region"                # region | cell | glyph | field | control
    role: Optional[str] = None          # VLM-assigned semantics
    parent: Optional[str] = None
    children: list = field(default_factory=list)
    resolution: str = "coarse"          # coarse | resolved | verified
    confidence: float = 0.5
    provenance: str = "measured"        # measured | vlm | corrected
    orient: Optional[str] = None
    needs_inspection: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("id", "bbox", "click", "kind", "role", "parent", "children",
                 "resolution", "confidence", "provenance", "orient",
                 "needs_inspection", "notes")}


class SceneState:
    """Canonical scene graph + refinement protocol.  The VLM drives quality
    (tighten/correct/relabel/dismiss), DETAIL (resolve), and RECOVERY (inspect)."""

    def __init__(self, frame_id: int = 0):
        self.entities: dict = {}
        self.events: list = []
        self.frame_id = int(frame_id)
        self.last_click_report: Optional[dict] = None   # what the last click CHANGED
        self.active_skill: Optional[str] = None         # the high-level strategy recipe in play

    def note_click_report(self, report: dict) -> None:
        """Record the substrate's post-click change report (what the click changed,
        incl. transient effects) so the VLM + live views see it."""
        self.last_click_report = report
        if isinstance(report, dict):
            self._log("click_report", summary=report.get("summary"))

    # ---- log + access -----------------------------------------------------
    def _log(self, op: str, **kw) -> None:
        self.events.append({"op": op, **kw})

    def add(self, ent: Entity) -> Entity:
        self.entities[ent.id] = ent
        return ent

    def get(self, eid: str) -> Optional[Entity]:
        return self.entities.get(eid)

    def click_for(self, eid: str) -> Optional[str]:
        """Click-by-id: the substrate's MEASURED centre for an id -> CLICK:col,row.
        The VLM references the id; the substrate owns the coordinate."""
        e = self.entities.get(eid)
        if not e or not e.click:
            return None
        return f"CLICK:{int(e.click[0])},{int(e.click[1])}"

    # ---- build (substrate-measured coarse entities) -----------------------
    def ingest_measured(self, frame_rgb, regions) -> list:
        """Seed COARSE top-level entities from measured regions.  ``regions`` is a
        list of {id, bbox, kind?, role?}; geometry is re-measured (tightened) from
        the frame so the bbox is the substrate's, not an eyeball.  A 'field'/large
        background region is kept coarse and never auto-resolved."""
        added = []
        for r in (regions or []):
            bb = [int(v) for v in r["bbox"]]
            cl = [int((bb[1] + bb[3]) // 2), int((bb[0] + bb[2]) // 2)]
            e = Entity(id=r["id"], bbox=bb, click=cl, kind=r.get("kind", "region"),
                       role=r.get("role"), provenance="measured")
            self.add(e)
            added.append(e.id)
        self._log("ingest_measured", ids=added)
        return added

    # ---- refinement ops (the protocol) ------------------------------------
    def resolve(self, eid: str, frame_rgb, into: str = "grid") -> list:
        """DETAIL on demand: decompose a coarse entity into addressable child bars/
        cells with stable ids (parent + r{r}c{c}) + measured clicks.  Lazy -- only
        what the VLM asks for is decomposed.  Returns the new child ids."""
        e = self.entities.get(eid)
        if e is None:
            return []
        kids = []
        if into == "grid":
            g = measure_grid(frame_rgb, e.bbox)
            if g:
                for b in g["bars"]:
                    r, c = b["rc"]
                    cid = f"{eid}/r{r}c{c}"
                    self.add(Entity(id=cid, bbox=b["bbox"], click=b["click"],
                                    kind="cell", parent=eid, orient=b["orient"],
                                    provenance="measured"))
                    kids.append(cid)
        else:  # components
            for i, comp in enumerate(measure_components(frame_rgb, e.bbox)):
                cid = f"{eid}/c{i}"
                self.add(Entity(id=cid, bbox=comp["bbox"], click=comp["click"],
                                kind="cell", parent=eid, orient=comp["orient"],
                                provenance="measured"))
                kids.append(cid)
        e.children = kids
        e.resolution = "resolved"
        self._log("resolve", id=eid, into=into, children=kids)
        return kids

    def tighten(self, eid: str, frame_rgb) -> Optional[list]:
        """QUALITY: re-measure an entity's bbox from the frame (snap to the union of
        its foreground components) -- the substrate corrects an eyeballed box."""
        e = self.entities.get(eid)
        if e is None:
            return None
        comps = measure_components(frame_rgb, e.bbox)
        if not comps:
            e.needs_inspection = True
            self._log("tighten", id=eid, result="no_components_flag_inspect")
            return None
        r0 = min(c["bbox"][0] for c in comps); c0 = min(c["bbox"][1] for c in comps)
        r1 = max(c["bbox"][2] for c in comps); c1 = max(c["bbox"][3] for c in comps)
        e.bbox = [r0, c0, r1, c1]
        e.click = [int((c0 + c1) // 2), int((r0 + r1) // 2)]
        e.provenance = "measured"
        self._log("tighten", id=eid, bbox=e.bbox)
        return e.bbox

    def correct(self, eid: str, bbox=None, role=None) -> bool:
        """The VLM corrects a mis-extraction (after inspecting the image)."""
        e = self.entities.get(eid)
        if e is None:
            return False
        if bbox is not None:
            e.bbox = [int(v) for v in bbox]
            e.click = [int((e.bbox[1] + e.bbox[3]) // 2), int((e.bbox[0] + e.bbox[2]) // 2)]
        if role is not None:
            e.role = role
        e.provenance = "corrected"
        e.needs_inspection = False
        self._log("correct", id=eid, bbox=bbox, role=role)
        return True

    def relabel(self, eid: str, role: str) -> bool:
        e = self.entities.get(eid)
        if e is None:
            return False
        e.role = role
        e.provenance = "vlm"
        self._log("relabel", id=eid, role=role)
        return True

    def verify(self, eid: str) -> bool:
        """The VLM confirms an entity is faithful (state matches the image)."""
        e = self.entities.get(eid)
        if e is None:
            return False
        e.resolution = "verified" if e.resolution != "resolved" else e.resolution
        e.needs_inspection = False
        e.confidence = max(e.confidence, 0.9)
        self._log("verify", id=eid)
        return True

    def flag_inspect(self, eid: str, why: str = "") -> bool:
        """Mark an entity as possibly-mis-extracted -> the VLM should inspect the
        image crop and recover.  The symbolic state is fallible."""
        e = self.entities.get(eid)
        if e is None:
            return False
        e.needs_inspection = True
        self._log("flag_inspect", id=eid, why=why)
        return True

    def dismiss(self, eid: str) -> bool:
        if eid not in self.entities:
            return False
        for cid in list(self.entities[eid].children):
            self.dismiss(cid)
        parent = self.entities[eid].parent
        if parent and parent in self.entities:
            self.entities[parent].children = [c for c in self.entities[parent].children if c != eid]
        del self.entities[eid]
        self._log("dismiss", id=eid)
        return True

    # ---- serialisation (for views + archive) ------------------------------
    def snapshot(self) -> dict:
        return {"frame_id": self.frame_id,
                "entities": {k: e.to_dict() for k, e in self.entities.items()},
                "events": list(self.events),
                "last_click_report": self.last_click_report,
                "active_skill": self.active_skill,
                "needs_inspection": [k for k, e in self.entities.items() if e.needs_inspection]}
