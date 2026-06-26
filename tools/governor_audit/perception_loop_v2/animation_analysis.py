"""Standalone animation entity-analysis — the SHARED implementation the live
driver runs, factored out so it can be unit-tested on saved animation frames
WITHOUT playing a game.

WHY THIS EXISTS
---------------
The exploratory driver analyses each action's animation frame-stack to surface
movement FACTS to the acting VLM: per-frame entities (palette-invariant), entity
movement events (object constancy across sub-frames), shape-identified moving
silhouettes, scene cuts (view changes), and a DEMONSTRATION/PREVIEW narration.
That logic used to live as private methods on ExploratoryDriver, so the only way
to exercise it was to play a whole game — slow, and impossible to inspect frame
by frame.

These free functions ARE that logic. The driver now delegates to them, so a unit
test calls the EXACT same code the agent uses in the loop (no duplicate to drift)
and can run it on any folder of saved frames. This is the foundation for the
animation unit-test harness (anim_unit_test.py) and for collecting animation
datasets for later use.

DESIGN
------
- Pure: no driver, no harness, no game adapter. The two pieces of state the
  driver methods used (the known entities {name: bbox} and the per-action
  demonstration cache) are passed in as parameters.
- Palette-invariant throughout (Adversarial Test / P11): figure-ground is decided
  by STRUCTURE via silhouette_track.foreground_components — never "the modal
  colour is background". Movers are ranked by CHANGE, not by colour.
- The substrate MEASURES (motion, extent, scene cuts); any IDENTITY or "what it
  controls" is emitted as a HYPOTHESIS with sub-1.0 credence for the VLM to
  verify — never asserted as fact.
- Fully guarded: every entry point degrades to an empty/neutral result rather
  than raising, exactly as the driver methods did.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import numpy as np
    from PIL import Image, ImageDraw
    _OK = True
except Exception:  # pragma: no cover - environment without numpy/PIL
    _OK = False


# ---------------------------------------------------------------------------
# Frame loading (shared) — every entry point normalises to 64x64 RGB int arrays.
# ---------------------------------------------------------------------------
def _load_frame(path) -> "np.ndarray":
    im = Image.open(path).convert("RGB")
    if im.size != (64, 64):
        im = im.resize((64, 64), Image.NEAREST)
    return np.asarray(im, dtype=int)


def frame_files(anim_dir) -> list:
    """Sorted frame_*.png paths in an animation directory ([] if none/missing)."""
    if anim_dir is None:
        return []
    return sorted(Path(str(anim_dir)).glob("frame_*.png"))


def load_frames(anim_dir) -> list:
    """Load an animation directory's frames as 64x64 RGB int arrays."""
    if not _OK:
        return []
    return [_load_frame(f) for f in frame_files(anim_dir)]


# ---------------------------------------------------------------------------
# 1. Colour-region motion summary (Layer-2)  [was _substrate_animation_summary]
# ---------------------------------------------------------------------------
def animation_summary(frame_stack) -> str:
    """Narrate the MOTION across an action's animation framestack so the VLM
    reads the effect's DYNAMICS (how things moved / grew / spread / flowed), not
    just the settled end state.  Tracks each foreground colour's extent across the
    sub-frames and reports its start -> peak -> end trajectory, ranking colours by
    how much they change (no fixed cutoff).  Palette-invariant (keys on CHANGE,
    not colour).  Pure; guarded.  ``frame_stack`` is a list of 2D-or-3D arrays;
    here we use the per-colour packed-int view the driver passes (a 2D label
    frame).  Accepts RGB frames too (packs them to ints)."""
    if not _OK:
        return "(animation tracker unavailable: numpy/PIL missing)"
    try:
        if not frame_stack or len(frame_stack) <= 1:
            return ("(no animation: the action produced a single settled "
                    "frame — no intermediate motion to read.)")
        frames = [_as_label_frame(g) for g in frame_stack]
        n = len(frames)
        colours = set()
        for g in frames:
            colours.update(int(v) for v in np.unique(g))

        def ext(g, c):
            m = (g == c)
            if not m.any():
                return None
            rs = np.where(m.any(axis=1))[0]
            cs = np.where(m.any(axis=0))[0]
            return (int(rs.min()), int(cs.min()), int(rs.max()),
                    int(cs.max()), int(m.sum()))

        def _cen(e):
            return ((e[0] + e[2]) / 2.0, (e[1] + e[3]) / 2.0) if e else None
        scored = []
        for c in colours:
            exts = [ext(g, c) for g in frames]
            change = 0.0
            for i in range(1, n):
                a, bb = exts[i - 1], exts[i]
                change += abs((bb[4] if bb else 0) - (a[4] if a else 0))
                pa, pb = _cen(a), _cen(bb)
                if pa and pb:
                    change += abs(pa[0] - pb[0]) + abs(pa[1] - pb[1])
                elif (pa is None) != (pb is None):
                    change += 5.0
            if change > 0:
                scored.append((change, c, exts))
        scored.sort(key=lambda t: t[0], reverse=True)
        if not scored:
            return (f"(animation: {n} frames, but no colour region changed "
                    f"position or extent across them.)")

        def b(e):
            return (f"bbox({e[0]},{e[1]},{e[2]},{e[3]}) {e[4]}px"
                    if e else "absent")

        def _runs(vals):
            runs = []
            for v in sorted(set(int(x) for x in vals)):
                if runs and v == runs[-1][1] + 1:
                    runs[-1][1] = v
                else:
                    runs.append([v, v])
            return [(a, b_) for a, b_ in runs]

        def _frontier_traj(c, e0, ep):
            if not (e0 and ep):
                return None
            up, down = e0[0] - ep[0], ep[2] - e0[2]
            left, right = e0[1] - ep[1], ep[3] - e0[3]
            best = max(up, down, left, right)
            if best <= 2:
                return None
            direction = ("up" if best == up else "down" if best == down
                         else "left" if best == left else "right")
            axis = "cols" if direction in ("up", "down") else "rows"
            seq = []
            for g in frames:
                m = (g == c)
                if not m.any():
                    continue
                rs = np.where(m.any(axis=1))[0]
                cs = np.where(m.any(axis=0))[0]
                if direction == "up":
                    edge = int(rs.min()); runs = _runs(np.where(m[edge])[0])
                elif direction == "down":
                    edge = int(rs.max()); runs = _runs(np.where(m[edge])[0])
                elif direction == "left":
                    edge = int(cs.min()); runs = _runs(np.where(m[:, edge])[0])
                else:
                    edge = int(cs.max()); runs = _runs(np.where(m[:, edge])[0])
                if not seq or seq[-1][1] != runs:
                    seq.append((edge, runs))
            return (direction, axis, seq)

        lines = [f"animation of {n} frames; colour regions that moved or "
                 f"changed (most-changed first):"]
        for _, c, exts in scored[:6]:
            e0, eN = exts[0], exts[-1]
            cells = [e[4] if e else 0 for e in exts]
            pk = int(np.argmax(cells)); ep = exts[pk]
            verb = []
            if e0 and ep:
                if ep[2] - e0[2] > 2 and abs(ep[0] - e0[0]) <= 2:
                    verb.append("extended DOWNWARD (top fixed, bottom advanced)")
                if (ep[3] - ep[1]) - (e0[3] - e0[1]) > 2:
                    verb.append("SPREAD horizontally")
                if e0[0] - ep[0] > 2:
                    verb.append("extended UPWARD")
                if ep[4] > e0[4] + 2:
                    verb.append("grew")
            if e0 and eN and abs(eN[4] - e0[4]) <= max(2, e0[4] // 3):
                dr = int(round(_cen(eN)[0] - _cen(e0)[0]))
                dc = int(round(_cen(eN)[1] - _cen(e0)[1]))
                if abs(dr) > 2 or abs(dc) > 2:
                    verb.append(f"MOVED dr={dr} dc={dc}")
            if e0 is None and ep is not None:
                verb.append("APPEARED mid-animation")
            if eN is None and e0 is not None:
                verb.append("ended ABSENT (left the frame / consumed)")
            vp = "; ".join(verb) or "changed extent"
            lines.append(f"- colour {c}: start {b(e0)} -> peak[f{pk}] "
                         f"{b(ep)} -> end {b(eN)}  [{vp}]")
            traj = _frontier_traj(c, e0, ep)
            if traj:
                direction, axis, seq = traj
                branched = any(len(runs) > 1 for _, runs in seq)
                pos = "row" if direction in ("up", "down") else "col"
                show = seq if len(seq) <= 9 else seq[:6] + [("...", [])] + seq[-2:]
                steps = " -> ".join(
                    ("..." if e == "..." else f"{pos}{e}{runs}")
                    for e, runs in show)
                tag = " (frontier SPLIT into branches)" if branched else ""
                lines.append(f"    leading edge ({direction}) over frames: "
                             f"{steps}{tag}")
        return "\n".join(lines)
    except Exception as e:
        return f"(animation tracker unavailable: {e})"


def _as_label_frame(g) -> "np.ndarray":
    """Normalise a frame to a 2D packed-int label frame.  A 3D RGB array is
    packed (r<<16|g<<8|b); a 2D array is used as-is."""
    a = np.asarray(g, dtype=int)
    if a.ndim == 3 and a.shape[2] == 3:
        return (a[:, :, 0] << 16) | (a[:, :, 1] << 8) | a[:, :, 2]
    return a


# ---------------------------------------------------------------------------
# 2. Entity-level movement events (Layer-3/4)  [was _substrate_animation_entities]
# ---------------------------------------------------------------------------
def animation_entities(frames) -> tuple:
    """ENTITY-LEVEL logical representation ACROSS an action's animation.

    Runs the SAME Frame->Entities extraction (palette-invariant per-colour
    components via silhouette_track.foreground_components) on EACH sub-frame,
    matches entities across consecutive frames by object constancy (dominant
    colour + nearest centroid; a co-located colour change = recolor-in-place),
    threads per-entity tracks, and emits MOVEMENT EVENTS (appeared / moved <dir>
    / grew / shrank / recoloured / vanished), each with its trajectory + net
    displacement + path span.  The substrate DETECTS the movement as a FACT; the
    acting VLM discovers the CORRELATION.  Game-agnostic; guarded.

    ``frames`` is a list of 64x64 RGB int arrays.  Returns (events, narration);
    events is a list of dicts, narration is the prompt text."""
    if not _OK:
        return [], "(animation entity tracker unavailable: numpy/PIL missing)"
    try:
        import silhouette_track as _ST_fg
        if not frames or len(frames) < 2:
            return [], "(no animation: single settled frame.)"
        frames = [np.asarray(f, dtype=int) for f in frames]

        # SCENE-CUT GUARD: a majority-of-frame repaint is a view change; tracking
        # across it invents phantom movers.  Restrict to the first contiguous
        # segment and warn the VLM to read the later phase directly.
        _cut_note = ""
        try:
            _npx = frames[0].shape[0] * frames[0].shape[1]
            _cut = next((i for i in range(1, len(frames))
                         if ((frames[i] != frames[i - 1]).any(axis=2)).sum() > 0.5 * _npx),
                        None)
            if _cut is not None:
                _cut_note = (f" (NOTE: tracked only sub-frames 0-{_cut - 1}; the "
                             f"animation CHANGES VIEW at sub-frame {_cut} -- read "
                             f"the filmstrip for the later phase, do not trust "
                             f"motion across the cut)")
                frames = frames[:_cut]
                if len(frames) < 2:
                    return [], "(animation changes view immediately; read the filmstrip.)"
        except Exception:
            pass

        def ents(arr):
            out = []
            for _e in _ST_fg.foreground_components(arr):
                r0, c0, r1, c1 = _e["bbox"]
                out.append({"cen": _e["centroid"],
                            "bbox": (int(r0), int(c0), int(r1), int(c1)),
                            "npix": _e["npix"], "colour": _e["colour"]})
            return out

        def cen(e):
            return e["cen"]
        def bbx(e):
            return e["bbox"]
        def npix(e):
            return e["npix"]
        def colour(e):
            return e["colour"]
        def hexc(c):
            return f"#{c:06X}" if isinstance(c, int) and c >= 0 else "?"

        def keep(L):
            return [e for e in L if npix(e) >= 5
                    and (bbx(e)[2] - bbx(e)[0]) < 40
                    and (bbx(e)[3] - bbx(e)[1]) < 40]

        per_frame = [keep(ents(f)) for f in frames]
        n = len(per_frame)

        def _size(e):
            r0, c0, r1, c1 = bbx(e)
            return (r1 - r0, c1 - c0)
        cohort = []
        for fr in per_frame:
            cc = {}
            for e in fr:
                cc[(colour(e), _size(e))] = cc.get((colour(e), _size(e)), 0) + 1
            cohort.append(cc)

        def match(A, B):
            cA = [colour(e) for e in A]; cB = [colour(e) for e in B]
            pA = [cen(e) for e in A];    pB = [cen(e) for e in B]
            cand = []
            for i in range(len(A)):
                for j in range(len(B)):
                    d = abs(pA[i][0]-pB[j][0]) + abs(pA[i][1]-pB[j][1])
                    same = cA[i] == cB[j]
                    if same or d <= 2:
                        cand.append((0 if same else 1, d, i, j))
            cand.sort()
            ai, bj, pairs = set(), set(), []
            for _pri, d, i, j in cand:
                if i in ai or j in bj:
                    continue
                ai.add(i); bj.add(j); pairs.append((i, j))
            appeared = [j for j in range(len(B)) if j not in bj]
            vanished = [i for i in range(len(A)) if i not in ai]
            return pairs, appeared, vanished

        tracks = []
        owner = {}
        for idx, e in enumerate(per_frame[0]):
            tracks.append({"frames": [0], "ent": [e]})
            owner[idx] = len(tracks) - 1
        for k in range(1, n):
            pairs, appeared, _vanished = match(per_frame[k-1], per_frame[k])
            new_owner = {}
            for i, j in pairs:
                tid = owner.get(i)
                if tid is None:
                    tracks.append({"frames": [k-1], "ent": [per_frame[k-1][i]]})
                    tid = len(tracks) - 1
                tracks[tid]["frames"].append(k)
                tracks[tid]["ent"].append(per_frame[k][j])
                new_owner[j] = tid
            for j in appeared:
                tracks.append({"frames": [k], "ent": [per_frame[k][j]]})
                new_owner[j] = len(tracks) - 1
            owner = new_owner

        events = []
        for tr in tracks:
            fs, es = tr["frames"], tr["ent"]
            c0, cN = cen(es[0]), cen(es[-1])
            dr, dc = cN[0]-c0[0], cN[1]-c0[1]
            man = abs(dr) + abs(dc)
            p0, pN = npix(es[0]), npix(es[-1])
            col0, colN = colour(es[0]), colour(es[-1])
            appeared = fs[0] > 0
            vanished = fs[-1] < n - 1
            uniq = (cohort[fs[0]].get((col0, _size(es[0])), 0) <= 1
                    and cohort[fs[-1]].get((colN, _size(es[-1])), 0) <= 1)
            pts = [cen(e) for e in es]
            exc, peak = 0.0, pts[0]
            for p in pts:
                d = abs(p[0] - pts[0][0]) + abs(p[1] - pts[0][1])
                if d > exc:
                    exc, peak = d, p

            def _dir(ddr, ddc):
                v = "down" if ddr > 1 else "up" if ddr < -1 else ""
                h = "right" if ddc > 1 else "left" if ddc < -1 else ""
                return "-".join(x for x in (v, h) if x) or "in place"

            verbs = []
            transient = False
            if appeared and uniq:
                verbs.append("appeared")
            if man > 2 and uniq:
                verbs.append("moved " + _dir(dr, dc))
            elif exc > 2 and uniq:
                transient = True
                verbs.append("moved " + _dir(peak[0]-pts[0][0], peak[1]-pts[0][1])
                             + " then RETURNED (transient mover -- net ~0)")
            if p0 and abs(pN - p0) > max(3, 0.25 * p0) and uniq:
                verbs.append("grew" if pN > p0 else "shrank")
            if col0 != colN:
                verbs.append("recoloured")
            if vanished and uniq:
                verbs.append("vanished")
            if not verbs:
                continue
            mag = max(man, exc) + abs(pN - p0) + (5 if appeared else 0) + (5 if vanished else 0)
            traj = [(fs[t], (round(cen(es[t])[0], 1), round(cen(es[t])[1], 1)),
                     npix(es[t])) for t in range(len(es))]
            rr = [pt[1][0] for pt in traj]; ccp = [pt[1][1] for pt in traj]
            events.append({
                "colour": colN, "colour_hex": hexc(colN),
                "size": f"{bbx(es[-1])[2]-bbx(es[-1])[0]}x{bbx(es[-1])[3]-bbx(es[-1])[1]}",
                "verbs": verbs,
                "first_frame": fs[0], "last_frame": fs[-1],
                "from": (round(c0[0], 1), round(c0[1], 1)),
                "to": (round(cN[0], 1), round(cN[1], 1)),
                "net": (round(dr, 1), round(dc, 1)),
                "path_span_rows": (round(min(rr), 1), round(max(rr), 1)),
                "path_span_cols": (round(min(ccp), 1), round(max(ccp), 1)),
                "trajectory": traj, "_mag": mag,
            })
        events.sort(key=lambda e: e["_mag"], reverse=True)
        for e in events:
            e.pop("_mag", None)

        if not events:
            return [], (f"animation of {n} frames{_cut_note}; the entity extractor "
                        f"found NO entity that moved/appeared/vanished across the "
                        f"sub-frames (motion may be sub-entity -- consult the "
                        f"filmstrip and the colour-region summary).")

        lines = [f"animation of {n} frames{_cut_note}; ENTITY movements detected by "
                 f"the substrate (object constancy across sub-frames -- each line "
                 f"is a FACT; YOU interpret what it correlates with):"]
        for e in events[:6]:
            path = " ".join(f"f{fr}{pt}" for fr, pt, _ in e["trajectory"])
            lines.append(
                f"- entity[colour {e['colour_hex']}, ~{e['size']}]: "
                f"{', '.join(e['verbs'])}; from {e['from']} to {e['to']} over "
                f"frames {e['first_frame']}..{e['last_frame']}, net {e['net']}; "
                f"path spans rows {e['path_span_rows']} cols {e['path_span_cols']}")
            lines.append(f"    path: {path}")
        if len(events) > 6:
            lines.append(f"  (+{len(events)-6} more lower-magnitude movements)")
        return events, "\n".join(lines)
    except Exception as e:
        return [], f"(animation entity tracker unavailable: {e})"


# ---------------------------------------------------------------------------
# 3. Shape-identified moving silhouettes  [was _substrate_silhouette_track]
# ---------------------------------------------------------------------------
def silhouette_movers(frames, known_entities: Optional[dict] = None) -> list:
    """Track a MOVING SILHOUETTE by NOVELTY (diff vs the static first sub-frame)
    and identify it with a known static entity by SCALE-NORMALISED SHAPE — so a
    grey silhouette tracing a path over a grey board (which the per-colour
    extractor merges into the background) is recovered AND recognised as e.g. the
    (abstracted) arch.  ``known_entities`` is {name: bbox(r0,c0,r1,c1)} (the
    driver passes its world.entities' current bboxes).  Returns identified movers
    ([] on miss).  Guarded."""
    if not _OK:
        return []
    try:
        import silhouette_track as _ST
        if not frames or len(frames) < 2:
            return []
        frames = [np.asarray(f, dtype=int) for f in frames]
        ents = dict(known_entities or {})
        known = _ST.entity_shape_masks(frames[0], ents)
        return _ST.track_silhouette(frames, known)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 4. Scene cuts (view changes)  [was _substrate_scene_cuts]
# ---------------------------------------------------------------------------
def scene_cuts(frames) -> list:
    """Frame indices where the animation CHANGES VIEW (a majority-of-frame
    repaint: a zoom / overlay / different screen).  Motion analysis is unreliable
    across such a cut, so they must be surfaced.  Returns a list of cut indices
    ([] none).  Guarded."""
    if not _OK:
        return []
    try:
        import silhouette_track as _ST
        if not frames or len(frames) < 2:
            return []
        return _ST.scene_cuts([np.asarray(f, dtype=int) for f in frames])
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 5. DEMONSTRATION / PREVIEW narration  [was _demonstration_narration]
# ---------------------------------------------------------------------------
def demonstration_narration(anim_events, last_action, silhouettes=None,
                            store: Optional[dict] = None) -> str:
    """Recognise a DEMONSTRATION/PREVIEW from the animation (game-agnostic).

    An entity that traces a path and RETURNS to ~its start (net displacement much
    smaller than the path it covered) leaves the settled frame unchanged yet is
    NOT inert — it PREVIEWED a motion.  Measures the previewed displacement in
    ticks so the acting VLM can match it to the transformation its goal needs.
    If ``store`` is provided, records demos on store[str(last_action)] (the driver
    passes its self._demonstrations cache).  Returns a narration string ('' when
    there is no demonstration).  Guarded."""
    try:
        demos = []
        for ev in (anim_events or []):
            traj = ev.get("trajectory") or []
            frm = ev.get("from")
            if frm is None or len(traj) < 2:
                continue
            r0, c0 = float(frm[0]), float(frm[1])
            far = max(traj, key=lambda t: (t[1][0] - r0) ** 2 + (t[1][1] - c0) ** 2)
            dr, dc = far[1][0] - r0, far[1][1] - c0
            reach = abs(dr) + abs(dc)
            net = ev.get("net") or [traj[-1][1][0] - r0, traj[-1][1][1] - c0]
            ret = abs(net[0]) + abs(net[1])
            if reach < 2.0 or ret > reach / 2.0:
                continue
            vdir = ("up" if dr < 0 else "down") if abs(dr) >= abs(dc) \
                else ("left" if dc < 0 else "right")
            demos.append({"colour": ev.get("colour_hex"), "size": ev.get("size"),
                          "dir": vdir, "d_row": round(dr, 1), "d_col": round(dc, 1),
                          "ticks": round(max(abs(dr), abs(dc)), 1)})
        sil_lines = []
        for s in (silhouettes or []):
            rr = s.get("reach", (0, 0))
            ticks = round(max(abs(rr[0]), abs(rr[1])), 1)
            if ticks < 2:
                continue
            iou = float(s.get("iou") or 0.0)
            demos.append({"colour": "shape:" + s.get("identity", "?"),
                          "size": "", "dir": s.get("dir"),
                          "d_row": rr[0], "d_col": rr[1], "ticks": ticks,
                          "identity": s.get("identity"),
                          "identity_credence": round(iou, 2)})
            conf = "LOW-confidence" if iou < 0.66 else "tentative"
            sil_lines.append(
                f"  - MEASURED: a silhouette PREVIEWED moving {s.get('dir')} "
                f"~{ticks} ticks (Δrow {rr[0]}, Δcol {rr[1]}).  HYPOTHESIS "
                f"({conf}, shape-match credence {round(iou, 2)}): it is the "
                f"'{s.get('identity')}' (abstracted) — a guess to VERIFY, not a fact.")
        if not demos:
            return ""
        if store is not None:
            store[str(last_action)] = demos
        lines = sil_lines + [
            f"  - MEASURED: entity[colour {d['colour']}, {d['size']}] PREVIEWED moving "
            f"{d['dir']} ~{d['ticks']} ticks (Δrow {d['d_row']}, Δcol {d['d_col']}) "
            f"then returned" for d in demos if not d.get("identity")]
        return ("[SUBSTRATE] DEMONSTRATION / PREVIEW — MEASURED FACT: this action's "
                "settled frame is ~unchanged but its animation TRACED a motion and "
                "returned, so the control is not inert and did not commit a change; it "
                "SHOWED a motion.  The motions below are measured; any IDENTITY or "
                "'what it controls' is a HYPOTHESIS to verify, not a fact.\n"
                + "\n".join(lines) +
                "\n  If a previewed motion matches the transformation your GOAL needs, "
                "treat 'this control/its settings are the answer' as a CLAIM to TEST "
                "(apply it, then check the score) at credence ~its shape-match — confirmed "
                "only by the outcome, never assumed.  Author it with sub-1.0 credence.")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 5b. SALIENT CO-OCCURRENCE -> a correlation claim binding two events
# ---------------------------------------------------------------------------
def _bbox_intersects(a, b) -> bool:
    """Whether two (r0,c0,r1,c1) bboxes intersect (INCLUSIVE -- a thin sweep line
    at row r still 'touches' an entity spanning that row, which an area test
    misses because a zero-height region has zero area)."""
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _resolve_entities(region, known_entities) -> list:
    """Named entities whose bbox INTERSECTS a region (r0,c0,r1,c1).  This GROUNDS
    a raw colour-blob event to the entities it actually touches -- e.g. a
    highlight sweep -> the switch entities it passed over -- so a claim names
    entities, not a skin-fragile colour token.  Sorted by column then row
    (reading order)."""
    out = []
    for nm, bb in (known_entities or {}).items():
        try:
            b = tuple(int(v) for v in bb)
        except Exception:
            continue
        if _bbox_intersects(region, b):
            out.append((b[1], b[0], nm))
    return [nm for _c, _r, nm in sorted(out)]


def _event_region(e):
    """The event's full SPATIAL extent over its path: the centroid span
    (path_span_rows/cols) EXPANDED by half the entity's size, so a sweeping bar
    (whose centroid-row is constant but whose body spans many rows) grounds to the
    entities its BODY crosses, not just the 1px centroid line."""
    pr = e.get("path_span_rows", (0, 0))
    pc = e.get("path_span_cols", (0, 0))
    size = str(e.get("size", "0x0"))
    try:
        sh, sw = (int(x) for x in size.split("x"))
    except Exception:
        sh, sw = 0, 0
    return (int(pr[0] - sh / 2), int(pc[0] - sw / 2),
            int(pr[1] + sh / 2), int(pc[1] + sw / 2))


def _salient_tracks(events, silhouettes, known_entities=None):
    """Normalise the animation's salient time-varying events into comparable
    tracks: each {label, frames(set), salience(ticks), target, grounded, kind}.
    Movers (identified silhouettes) and entity events are put on the same footing
    so the two most salient can be compared for co-occurrence.  ``grounded`` is
    the named entities the track overlaps (the silhouette's own identity, or the
    entities a colour event swept over) so the claim can name ENTITIES."""
    tracks = []
    for s in (silhouettes or []):
        fr = {int(k) for k, _ in (s.get("trajectory") or [])}
        if not fr:
            continue
        reach = s.get("reach", (0, 0))
        mag = max(abs(reach[0]), abs(reach[1]))
        ident = s.get("identity")
        tracks.append({
            "label": f"the '{ident}' silhouette moves "
                     f"{s.get('dir')} ~{round(mag, 1)} ticks"
                     + (f" (opening {s.get('opening')})" if s.get("opening") else ""),
            "frames": fr, "salience": float(mag),
            "target": ident, "grounded": [ident] if ident else [],
            "kind": "moving_silhouette"})
    for e in (events or []):
        f0, f1 = int(e.get("first_frame", 0)), int(e.get("last_frame", 0))
        fr = set(range(f0, f1 + 1))
        if not fr:
            continue
        pr = e.get("path_span_rows", (0, 0))
        pc = e.get("path_span_cols", (0, 0))
        net = e.get("net", (0, 0))
        sal = (abs(pr[1] - pr[0]) + abs(pc[1] - pc[0])) or (abs(net[0]) + abs(net[1]))
        grounded = _resolve_entities(_event_region(e), known_entities)
        over = (" over " + ", ".join(grounded)) if grounded else ""
        tracks.append({
            "label": f"entity[{e.get('colour_hex', '?')}] "
                     f"{', '.join(e.get('verbs', []))}{over}",
            "frames": fr, "salience": float(sal),
            "target": e.get("colour_hex"), "grounded": grounded,
            "kind": "entity_event"})
    return tracks


def salient_cooccurrence(events, silhouettes, known_entities=None):
    """Find the TWO most-salient events that CO-OCCUR across multiple animation
    frames -- a candidate CORRELATION worth binding with a claim.  Two salient
    things that change TOGETHER, repeatedly, are a designer's demonstration that
    one is bound to the other (e.g. a marker moving up WHILE the switches that
    control it highlight, affirmed every frame).

    Anchors on an identified moving SILHOUETTE (the goal-relevant marker) when
    present, binding it to its most-salient co-occurring event; else binds the
    top-2 co-occurring events.  ``known_entities`` ({name: bbox}) GROUNDS each
    event to the entities it touches, so the binding names ENTITIES (e.g. the
    switches a highlight swept), not a skin-fragile colour token.  Returns
    {a, b, co_frames, n_co, n_frames} or None.  The substrate MEASURES the
    co-occurrence (a fact); the causal binding is a HYPOTHESIS to verify."""
    tracks = _salient_tracks(events, silhouettes, known_entities)
    if len(tracks) < 2:
        return None
    n_frames = max((max(t["frames"]) for t in tracks), default=0) + 1
    sils = [t for t in tracks if t["kind"] == "moving_silhouette"]
    evs = [t for t in tracks if t["kind"] != "moving_silhouette"]
    pairs = []
    if sils and evs:                       # anchor on the identified marker
        for s in sils:
            for e in evs:
                common = s["frames"] & e["frames"]
                if len(common) >= 2:
                    pairs.append((s["salience"] + e["salience"], s, e, common))
    if not pairs:                          # fall back to top-2 of everything
        allt = sorted(tracks, key=lambda t: -t["salience"])
        for i in range(len(allt)):
            for j in range(i + 1, len(allt)):
                common = allt[i]["frames"] & allt[j]["frames"]
                if len(common) >= 2:
                    pairs.append((allt[i]["salience"] + allt[j]["salience"],
                                  allt[i], allt[j], common))
                    break
            if pairs:
                break
    if not pairs:
        return None
    pairs.sort(key=lambda p: -p[0])
    _sal, a, b, common = pairs[0]
    return {"a": {"label": a["label"], "target": a["target"],
                  "grounded": a.get("grounded", []), "kind": a["kind"]},
            "b": {"label": b["label"], "target": b["target"],
                  "grounded": b.get("grounded", []), "kind": b["kind"]},
            "co_frames": sorted(common), "n_co": len(common), "n_frames": n_frames}


def correlation_narration(cooc) -> str:
    """Prompt text surfacing a salient co-occurrence as a candidate correlation
    for the VLM to BIND with a claim.  Credence ~ the share of frames affirming
    it; the binding is explicitly a hypothesis to verify, not a fact."""
    if not cooc:
        return ""
    cred = round(cooc["n_co"] / max(1, cooc["n_frames"]), 2)
    return ("[SUBSTRATE] SALIENT CO-OCCURRENCE — MEASURED FACT: the two most "
            f"salient events in this animation CHANGE TOGETHER across "
            f"{cooc['n_co']} frame(s) {cooc['co_frames']}:\n"
            f"  (A) {cooc['a']['label']}\n"
            f"  (B) {cooc['b']['label']}\n"
            "Two salient things that co-vary, repeatedly, are a designer's CLUE "
            "that one is bound to the other.  Author a CORRELATION CLAIM binding "
            "A and B ('B controls/indicates A') and TEST it -- reproduce B (set "
            "the controls to the shown pattern) and check whether A and the score "
            f"follow.  Credence ~{cred} (the share of frames affirming the "
            "co-occurrence); the binding is a HYPOTHESIS to verify, not a fact.")


def correlation_claim(cooc, last_action) -> Optional[dict]:
    """A low-credence (GUESSED) correlation claim record for the Claim Store,
    binding the two co-occurring salient events.  ClaimStore.ingest clamps the
    credence to the GUESSED ceiling, so this never asserts a hardcoded fact -- it
    is a hypothesis the prober will TEST (reproduce B; check A + score)."""
    if not cooc:
        return None
    a, b = cooc["a"], cooc["b"]
    frac = cooc["n_co"] / max(1, cooc["n_frames"])
    # Prefer GROUNDED entity ids (the named entities each event touches) over the
    # raw colour/identity token -- so the claim points at ENTITIES (the switches a
    # highlight swept), survives a re-skin, and is re-locatable when verifying.
    targets = list(dict.fromkeys(
        (a.get("grounded") or [a.get("target")]) + (b.get("grounded") or [b.get("target")])))
    targets = [t for t in targets if t]
    a_tok = a.get("target") or "A"
    b_tok = "+".join(b.get("grounded") or []) or b.get("target") or "B"
    cid = f"corr::{last_action}::{a_tok}~{b_tok}"
    return {
        "id": cid,
        "statement": (f"CO-OCCURRENCE (preview {last_action}): {a['label']} "
                      f"co-varies with {b['label']} across {cooc['n_co']}/"
                      f"{cooc['n_frames']} frames -> candidate correlation "
                      "(B controls/indicates A); TEST by reproducing B and "
                      "checking A + the score."),
        "kind": "correlation", "scope": "level", "target": targets,
        "plan": (f"reproduce '{b['label']}' (apply the shown control pattern), "
                 f"then check whether '{a['label']}' and the score follow"),
        "provenance": "guessed", "credence": round(min(frac, 0.40), 2),
        "importance": 0.7, "cost": 2,
    }


# ---------------------------------------------------------------------------
# 6. Per-frame entity-analysis filmstrip render  [was _render_entity_analysis_filmstrip]
# ---------------------------------------------------------------------------
def render_entity_analysis_filmstrip(frames, out_path, known_entities=None):
    """Render each animation sub-frame with the substrate's PER-FRAME ENTITY
    ANALYSIS drawn ON it — every detected entity's bbox + a label (id, colour,
    size).  This is the verification view: it shows EXACTLY what the entity
    extractor found in each animated frame.  A scene cut (majority repaint) is
    marked.  YELLOW = moved/re-detected this frame (box FOLLOWS the entity);
    GREEN = unchanged (carried).  ``frames`` is a list of 64x64 RGB int arrays.
    Pure rendering; guarded; returns out_path or None."""
    if not _OK:
        return None
    try:
        import silhouette_track as _ST
        from substrate_tools.frameutils import font as _ft_font
        if not frames or len(frames) < 2:
            return None
        frames = [np.asarray(f, dtype=int) for f in frames]

        tracked = _ST.track_per_frame_entities(frames)
        cuts = set(_ST.scene_cuts(frames))

        def _ov(a, b):
            r0, c0 = max(a[0], b[0]), max(a[1], b[1])
            r1, c1 = min(a[2], b[2]), min(a[3], b[3])
            return max(0, r1 - r0 + 1) * max(0, c1 - c0 + 1)

        known = {}
        for nm, bb in (known_entities or {}).items():
            try:
                known[nm] = tuple(int(v) for v in bb)
            except Exception:
                continue
        id_name = {}
        for e in (tracked[0] if tracked else []):
            best, bn = 0, None
            for nm, bb in known.items():
                ov = _ov(e["bbox"], bb)
                if ov > best:
                    best, bn = ov, nm
            if bn:
                id_name[e["id"]] = bn

        up = 7
        cols = min(len(frames), 5)
        rows = (len(frames) + cols - 1) // cols
        tw = th = 64 * up
        lab, gap = 46, 8
        W = cols * tw + (cols + 1) * gap
        H = rows * (th + lab) + (rows + 1) * gap
        canvas = Image.new("RGB", (W, H), (18, 18, 24))
        d = ImageDraw.Draw(canvas)
        fnt = _ft_font(15)
        fsm = _ft_font(12)
        d.text((gap, 1), "CHANGE-DRIVEN per-frame entities (palette-invariant): "
               "YELLOW = moved / re-detected this frame (box FOLLOWS the entity); "
               "GREEN = unchanged (carried, same entity); label = id:name #colour. "
               "Texture/fields suppressed.", fill=(200, 200, 160), font=fsm)
        for i, arr in enumerate(frames):
            tile = Image.fromarray(np.asarray(arr, dtype="uint8"), "RGB").resize(
                (tw, th), Image.NEAREST)
            td = ImageDraw.Draw(tile)
            for e in tracked[i]:
                r0, c0, r1, c1 = e["bbox"]
                col = (255, 210, 40) if e["moved"] else (60, 210, 90)
                td.rectangle((c0 * up, r0 * up, (c1 + 1) * up - 1, (r1 + 1) * up - 1),
                             outline=col, width=2 if e["moved"] else 1)
                nm = id_name.get(e["id"], "")
                lbl = f"{e['id']}:{nm}" if nm else f"{e['id']} #{e['colour']:06X}"
                td.text((c0 * up + 1, max(0, r0 * up - 12)), lbl, fill=col, font=fsm)
            r, c = divmod(i, cols)
            x = gap + c * (tw + gap)
            y = (gap + 16) + r * (th + lab + gap)
            nmoved = sum(1 for e in tracked[i] if e["moved"])
            cut = "  <SCENE CUT (view change)" if i in cuts else ""
            d.text((x + 2, y + 2),
                   f"frame {i}: {len(tracked[i])} entities ({nmoved} moved){cut}",
                   fill=(255, 140, 140) if cut else (240, 240, 140), font=fnt)
            canvas.paste(tile, (x, y + lab - 16))
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)
        return out_path
    except Exception as e:
        print(f"[animation] entity-analysis filmstrip render failed ({e}); skipping",
              flush=True)
        return None


# ---------------------------------------------------------------------------
# Top-level orchestrator — everything COS derives from an animation, in one call.
# ---------------------------------------------------------------------------
def analyze(anim_dir, known_entities: Optional[dict] = None,
            action: Optional[str] = None) -> dict:
    """Run the full animation analysis on a directory of saved frames and return
    a structured result (the entity FACTS + COS's interpretation).  This mirrors
    what the live driver derives per animated turn, so a unit test inspects the
    exact same outputs.

    Returns a dict:
      n_frames, scene_cuts,
      per_frame_entities : [[{id,bbox,colour,npix,moved}, ...], ...]
      events             : entity movement events (animation_entities)
      events_narration   : prompt text for the events
      colour_summary     : animation_summary text
      silhouettes        : shape-identified movers
      demonstration      : demonstration/preview narration ('' if none)
    """
    import silhouette_track as _ST
    frames = load_frames(anim_dir)
    if len(frames) < 2:
        return {"n_frames": len(frames), "scene_cuts": [], "per_frame_entities": [],
                "events": [], "events_narration": "(no animation)",
                "colour_summary": "(no animation)", "silhouettes": [],
                "demonstration": ""}
    tracked = _ST.track_per_frame_entities(frames)
    per_frame = [[{"id": e["id"], "bbox": list(e["bbox"]), "colour": e["colour"],
                   "npix": e["npix"], "moved": bool(e["moved"])} for e in fr]
                 for fr in tracked]
    act = action or str(anim_dir)
    events, events_narration = animation_entities(frames)
    sils = silhouette_movers(frames, known_entities)
    demo = demonstration_narration(events, act, sils, store=None)
    cooc = salient_cooccurrence(events, sils, known_entities)
    return {
        "n_frames": len(frames),
        "scene_cuts": scene_cuts(frames),
        "per_frame_entities": per_frame,
        "events": events,
        "events_narration": events_narration,
        "colour_summary": animation_summary(frames),
        "silhouettes": sils,
        "demonstration": demo,
        "cooccurrence": cooc,
        "correlation": correlation_narration(cooc),
        "correlation_claim": correlation_claim(cooc, act),
    }
