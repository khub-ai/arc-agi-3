"""Qwen3-VL-8B-thinking as the in-loop COS VLM for su15 lc0 (strict mode).

Watches the ExploratoryDriver work dir and answers its file-handoff prompts by
calling qwen/qwen3-vl-8b-thinking. Per SPEC_small_model_compensation, the SMALL
MODEL is used only on its RELIABLE axes (perceive, label, endorse one step) and
the SUBSTRATE owns the geometry/decomposition the benchmark shows it botches:

  - PERCEPTION: the substrate `components` op finds the blobs (accurate bboxes);
    Qwen only LABELS each blob's ROLE + names the game (its 2/2 perception axis).
    Qwen never has to emit precise coordinates (its weak localization).
  - STRATEGY: Qwen ENDORSES the mechanical actor's already-proposed action
    rather than authoring a plan (compensates means_ends 0/1, closed-vocab drift).
  - REFINEMENT/REQUERY gates: re-emit the cached, substrate-grounded perception
    (no Qwen re-derivation -> avoids its weak self_correction 0/1).
  - Reliability: thinking mode on; k=2 self-consistency on the role labelling.
"""
from __future__ import annotations
import sys, json, time, re, base64, glob, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]    # arc-agi-3
sys.path.insert(0, str(REPO / "usecases/arc-agi-3/python"))
sys.path.insert(0, str(REPO / "tools/governor_audit/perception_loop_v2"))
import backends
import substrate_tools.registry as reg

# WORK/RAW_DIR/MODEL are reconfigured by serve() for in-process competition use;
# these defaults keep the module runnable standalone (the old dev-trial responder).
WORK = REPO / ".tmp/sm_trial/su15_qwen"
RAW_DIR = REPO / ".tmp/exploratory_play_frames/su15"
# Model-agnostic: swap the VLM with COS_VLM_MODEL (any backends.call_oracle slug --
# anthropic / openai / openrouter / together). The bridge logic is identical for any model.
MODEL = os.environ.get("COS_VLM_MODEL", "qwen/qwen3-vl-8b-thinking")

# Gated CoT, VLM-AGNOSTIC.  Hard visual steps (perceive / read-animation) may run on a
# DEEP model; routine steps (endorse / menu / gates) run on a FAST model.  Both default
# to MODEL, so a single-model deployment is byte-for-byte unchanged; set DEEP via serve()
# or COS_DEEP_VLM_MODEL to escalate just the hard steps.  The slugs are arbitrary
# backends.call_oracle slugs (qwen instruct|thinking, anthropic, openai, ...) -- the
# routing never assumes a specific provider, so it works with any VLM.
FAST_MODEL = MODEL
DEEP_MODEL = os.environ.get("COS_DEEP_VLM_MODEL", MODEL)

_cost = {"calls": 0, "in": 0, "out": 0, "s": 0.0, "usd": 0.0}
_last_perception = {"entities": [], "game_type": "", "game_purpose": ""}
# SUBSTRATE-TRACKED STABLE IDS: persistent identity for measured components, matched across
# frames by colour + nearest position (threshold-free).  Each item carries the last VLM label
# so the same physical region keeps its name even when the VLM would otherwise re-name it --
# robust THROUGH scene change (a vanished region drops, a new one gets a fresh id).
_component_tracker = {"next_id": 0, "items": []}     # items: {id,color,center,bbox,label}


def _track_components(cands):
    """Assign stable ids to this frame's components by matching to the prior frame: within a
    colour, each current component takes its NEAREST unused prior component (greedy, no tuned
    distance threshold).  Matched -> carry id + prior label; unmatched -> new id.  Updates the
    tracker to the current frame and returns the per-cand tracking records (aligned to cands)."""
    prior = _component_tracker["items"]
    used, out = set(), []
    for b in cands:
        col = b.get("color") or ""
        bb = b["bbox"]; ctr = ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)
        best, bestd = None, None
        for j, p in enumerate(prior):
            if j in used or p["color"] != col:
                continue
            d = abs(p["center"][0] - ctr[0]) + abs(p["center"][1] - ctr[1])
            if bestd is None or d < bestd:
                best, bestd = j, d
        if best is not None:
            used.add(best)
            cid, label = prior[best]["id"], prior[best]["label"]
        else:
            cid, label = _component_tracker["next_id"], None
            _component_tracker["next_id"] += 1
        out.append({"id": cid, "color": col, "center": ctr, "bbox": bb, "label": label})
    _component_tracker["items"] = out
    return out


def _label_components(entities):
    """After the VLM names this frame's objects, attach each tracked component's label from the
    entity whose bbox contains its centre -- so next frame the SAME physical region offers that
    label back to the VLM (stable naming via stable id)."""
    for c in _component_tracker["items"]:
        r, cc_ = c["center"]
        for e in entities:
            bb = e.get("bbox_ticks_turn1")
            if (isinstance(bb, (list, tuple)) and len(bb) == 4
                    and bb[0] <= r <= bb[2] and bb[1] <= cc_ <= bb[3]):
                c["label"] = e.get("name")
                break


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _latest_raw():
    cands = [p for p in glob.glob(str(RAW_DIR / "turn_*.png")) if "anim" not in p]
    if not cands:
        return None
    return sorted(cands, key=lambda p: int(os.path.basename(p).split("turn_")[1].split(".")[0]))[-1]


def _b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode() if path else None


# LOCAL model (Ollama over LAN) takes priority when OLLAMA_URL is set -- a GPU-local
# Qwen has no API latency/throttle, and matches the competition setup (model on the
# same box as COS).  Native /api/chat with base64 images + think mode.
OLLAMA_URL = os.environ.get("OLLAMA_URL")                       # e.g. http://127.0.0.1:11434
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-vl:8b")
OLLAMA_THINK = os.environ.get("OLLAMA_THINK", "1") != "0"      # think mode; set 0 for fast picks
import urllib.request as _u


def _b64_small(path, maxpx=384):
    """Downscale to <=maxpx before encoding -- a 1270px annotated frame is ~1000
    image tokens; the VLM only needs to SEE the pieces (the substrate supplies the
    coordinates), so a smaller image cuts prompt-eval on a modest local GPU."""
    try:
        import io
        from PIL import Image
        im = Image.open(path).convert("RGB")
        if max(im.size) > maxpx:
            s = maxpx / max(im.size)
            im = im.resize((int(im.size[0] * s), int(im.size[1] * s)))
        buf = io.BytesIO(); im.save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return _b64(path)


def _ollama_call(system, user, image_path, max_tokens):
    # Qwen's /no_think soft switch ACTUALLY suppresses the reasoning trace (the
    # Ollama think:False flag alone does not for qwen3-vl -> it still thinks and
    # eats the token budget, returning empty content).  Append it when not thinking.
    if not OLLAMA_THINK:
        user = user + "\n/no_think"
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if image_path:
        msgs[1]["images"] = [_b64_small(image_path)]
    body = json.dumps({"model": OLLAMA_MODEL, "messages": msgs, "think": OLLAMA_THINK,
                       "stream": False,
                       "options": {"num_ctx": 8192, "num_gpu": 99, "temperature": 0.6,
                                   "top_p": 0.95, "num_predict": int(max_tokens)}}).encode()
    req = _u.Request(OLLAMA_URL + "/api/chat", data=body,
                     headers={"Content-Type": "application/json"})
    r = json.load(_u.urlopen(req, timeout=300))
    return (r.get("message", {}) or {}).get("content", "") or ""


def _qwen(system, user, image_path=None, max_tokens=1600, think=None, tier="fast"):
    t0 = time.time()
    slug = DEEP_MODEL if tier == "deep" else FAST_MODEL
    if OLLAMA_URL:
        global OLLAMA_THINK
        _save = OLLAMA_THINK
        if think is not None:
            OLLAMA_THINK = bool(think)
        for attempt in range(3):
            try:
                out = _ollama_call(system, user, image_path, max_tokens)
                OLLAMA_THINK = _save
                _cost["calls"] += 1; _cost["s"] += time.time() - t0
                return out
            except Exception:
                time.sleep(2 * (attempt + 1))
        OLLAMA_THINK = _save
        return ""
    for attempt in range(4):
        try:
            backends.check_budget()       # ArcPrize $10K cap: refuse before overspending
            res = backends.call_oracle(model=slug, system=system, user=user,
                                       image_b64=_b64(image_path), max_tokens=max_tokens,
                                       temperature=0.0, timeout_s=180)
            _cost["calls"] += 1
            _cost["in"] += int(res.get("input_tokens", 0) or 0)
            _cost["out"] += int(res.get("output_tokens", 0) or 0)
            if not res.get("_cache_hit"):     # cached replays cost nothing -- don't double-count
                _cost["usd"] = backends.record_spend(res.get("cost_usd", 0))
            _cost["s"] += time.time() - t0
            return res.get("reply") or res.get("text") or ""
        except backends.BudgetExceeded as e:
            # terminal: stop spending -- do NOT retry; let the run degrade/end.
            print(f"[cos-responder] {e}; halting VLM calls", flush=True)
            return ""
        except Exception:
            time.sleep(min(30, 4 * (2 ** attempt)))
    return ""


def _json_obj(txt):
    s = (txt or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s); s = re.sub(r"\n?```\s*$", "", s)
    start = s.find("{")
    while start != -1:
        depth = 0
        for j in range(start, len(s)):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:j + 1])
                    except Exception:
                        break
        start = s.find("{", start + 1)
    return None


def _blobs(raw_path):
    """Substrate-measured foreground units on the RAW 64x64 frame: accurate bboxes +
    centroids + colours (the geometry the VLM gets wrong), GROUPED into coherent
    objects via substrate_objects.extract_objects (figure-ground + per-colour grouping
    + texture suppression) -- NOT raw connected components, which over-split a dotted
    ring into N one-pixel dots and a line into shards, flood the candidate budget, and
    starve the VLM of real objects (durable principle P17).  Falls back to the raw
    components op only if extract_objects is unavailable."""
    try:
        import numpy as np
        from PIL import Image
        import substrate_objects as _so
        a = np.array(Image.open(raw_path).convert("RGB"))
        s = max(1, a.shape[0] // 64)
        g = a[s // 2::s, s // 2::s][:64, :64]          # clean native 64x64 (skip grid)
        out = []
        for o in _so.extract_objects(g):
            r0, c0, r1, c1 = o["bbox"]
            out.append({"bbox": [int(r0), int(c0), int(r1), int(c1)],
                        "center": [int(o["center"][0]), int(o["center"][1])],
                        "cells": int(o["npix"]), "color": o["color"]})
        if out:
            return out
    except Exception:
        pass
    try:                                               # fallback: raw components op
        r = reg.run_queries(raw_path, [{"op": "components", "id": "b",
                                        "bbox": [0, 0, 64, 64], "min_cells": 1,
                                        "max_return": 24}], "/tmp/qwenresp", n_ticks=64)
        comps = r[0].get("components") or []
    except Exception:
        comps = []
    out = []
    for c in comps:
        r0, c0, r1, c1 = c["bbox"]
        out.append({"bbox": [int(r0), int(c0), int(r1), int(c1)],
                    "center": [round(c["centroid"][0]), round(c["centroid"][1])],
                    "cells": c["cells"], "color": c["color"]})
    return out


# ---------------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------------
PERC_SYS = (
    "You are the PERCEPTION engine of a game-playing system. LOOK at the attached "
    "grid-annotated frame (the axes are TICK coordinates, row 0 = top, col 0 = left). "
    "CRITICAL COORDINATE CONTRACT: every bbox number you output MUST be a 0-63 TICK label "
    "(the numbers printed on the grid axes). The frame is ENLARGED ~10x for visibility, but "
    "you MUST report the SMALL tick numbers (0-63), NEVER image-pixel numbers (which reach "
    "~1270). Every coordinate is between 0 and 63; if you wrote a number above 63 you used "
    "pixels by mistake -- convert it to the tick label. "
    "IDENTIFY every distinct game object yourself -- the player/controllable "
    "piece, every movable piece, targets/goals, markers/cursors, walls/barriers, and "
    "small but distinct marks (a coloured centre, a slot). CRITICAL: a piece of a "
    "different colour from its surround is its OWN object, NEVER part of the field or "
    "room it sits on; do not miss small ones. The substrate's measured candidates "
    "(listed) are only UNRELIABLE HINTS -- it merges distinct pieces into a field and "
    "splits one piece into colour fragments, so CORRECT them (merge fragments of one "
    "object; ADD objects it missed). For each REAL object give a rough tick bbox "
    "[row0,col0,row1,col1] (the substrate refines it later, so approximate is fine), a "
    "ROLE, and a 2-3 word appearance.  Reply ONE JSON object, first char `{`, no prose.\n"
    "ROLES (pick one): mover (the thing the player controls/moves), goal (a target the "
    "mover should reach), marker (a small transient cue/cross/cursor), hud (a status "
    "bar/counter/legend), wall, decoration.\n"
    "JSON schema: {\"entities\":[{\"name\":<short_unique>,\"bbox\":[r0,c0,r1,c1],"
    "\"role\":<role>,\"appearance\":<2-3 words>}], \"game_type\":<short>, "
    "\"game_purpose\":<one sentence: what the player must do to win>}")


def _relation_facts(raw_path):
    """AUTHORITATIVE measured geometric relations among foreground components --
    identical-by-(shape AND size) and containment -- the reliable STRUCTURE the
    VLM anchors its naming on (a small VLM eyeballs a 64x64 grid and merges /
    miscounts pieces).  Pure geometry; the VLM still decides MEANING.  The
    same-shape-modulo-scale relation is dropped from the prompt (too promiscuous
    at tiny scale: every solid blob collapses to a filled square).  '' on any
    issue.  See substrate_relations + the measurement-vs-meaning line."""
    try:
        import numpy as np
        from PIL import Image
        import silhouette_track as st
        import substrate_relations as sr
        rgb = np.array(Image.open(raw_path).convert("RGB"))
        if rgb.shape[:2] != (64, 64):     # only the clean raw tick frame; never an
            return ""                      # upscaled/annotated image (anti-alias specks)
        comps = st.foreground_components(rgb)
        co = [{"id": i, "bbox": list(c["bbox"]), "mask": np.asarray(c["mask"]),
               "npix": int(np.asarray(c["mask"]).sum())} for i, c in enumerate(comps)]
        rel = sr.find_relations(co)
        if not rel:
            return ""
        rel = dict(rel)
        rel["same_shape"] = []
        def _ctr(bb):
            return ((bb[0] + bb[2]) // 2, (bb[1] + bb[3]) // 2)
        names = {c["id"]: f"shape@(row{_ctr(c['bbox'])[0]},col{_ctr(c['bbox'])[1]}),{c['npix']}px"
                 for c in co}
        return sr.render_text(rel, names)
    except Exception:
        return ""


def _pair_relations(raw_path, entities):
    """Surface the substrate's MEASURED compatible/complementary PAIRS -- identical
    (same kind / swap-pair candidate), shape_similar, and fits (a piece whose footprint
    fits a container/slot -- complementary) -- mapped onto the VLM's entity names, so
    the planner + swap idiom can use them.  Detecting two compatible/complementary
    entities is central to many ARC games; the substrate measures the geometry (facts),
    the planner/VLM interprets the meaning (P17).  [] on any issue."""
    try:
        import numpy as np
        from PIL import Image
        import substrate_objects as so
        import substrate_relations as sr
        rgb = np.array(Image.open(raw_path).convert("RGB"))
        if rgb.shape[:2] != (64, 64):           # downsample an upscaled frame (like _blobs)
            s = max(1, rgb.shape[0] // 64)
            rgb = rgb[s // 2::s, s // 2::s][:64, :64]
        if rgb.shape[:2] != (64, 64):
            return []
        objs = so.extract_objects(rgb)          # GROUPED objects (P17), not raw fragments
        rel = sr.find_relations(objs)
        if not rel:
            return []
        by_id = {o["id"]: o for o in objs}

        def name_of(oid):
            o = by_id.get(oid)
            if not o:
                return None
            cr0, cc0, cr1, cc1 = o["bbox"]
            best, nm = 0, None
            for e in entities:
                r0, c0, r1, c1 = e["bbox_ticks_turn1"]
                ov = (max(0, min(cr1, r1) - max(cr0, r0))
                      * max(0, min(cc1, c1) - max(cc0, c0)))
                if ov > best:
                    best, nm = ov, e["name"]
            return nm

        out, seen = [], set()

        def add(names, relation, note, ordered=False):
            # emit PAIRWISE {from,to,relation} (the world model ingests from/to; a
            # {between:[...]} record is silently dropped by ingest_perception).
            # ordered=True keeps direction (fits: from=piece, to=container).
            nm = [n for n in names if n] if ordered else sorted({n for n in names if n})
            for i in range(len(nm)):
                for j in range(i + 1, len(nm)):
                    key = (nm[i], nm[j], relation)
                    if key not in seen:
                        seen.add(key)
                        # 'evidence' (not 'note') -> ingest_perception stores it on the
                        # RelationshipRecord, so the pair detail shows on the trace.
                        out.append({"from": nm[i], "to": nm[j],
                                    "relation": relation, "evidence": note})

        for grp in rel.get("identical", []):
            add((name_of(i) for i in grp), "identical",
                "same shape+size -> same KIND / swap-pair candidate")
        for s in rel.get("shape_similar", []):
            add((name_of(s["a"]), name_of(s["b"])), "shape_similar",
                "same shape, differing -> complementary candidate")
        for f in rel.get("fits", []):
            add((name_of(f["piece"]), name_of(f["container"])), "fits",
                "the piece's footprint fits the container interior -> piece<->slot",
                ordered=True)        # from=piece, to=container
        return out
    except Exception:
        return []


def _shape_swap_candidates(raw_path, entities):
    """Swap-pair candidates: entity pairs that are the SAME SHAPE (silhouette match,
    rotation/scale-invariant) but a DIFFERENT colour -- the swap-pair signature.
    Grounded on SHAPE, not bbox size, so it does NOT repeat the size-congruence
    mis-pairing rejected for ka59.  Crops each entity by its (VLM-authored) bbox from
    the raw frame -- reliable, because it bypasses the full-frame extraction that merges
    the two r11l flowers -- and reuses the tested substrate_relations shape matcher.
    Substrate ASSIST only: it surfaces the candidate as EVIDENCE; the VLM decides whether
    they truly swap, and a probe validates the claim."""
    try:
        import numpy as np
        from PIL import Image
        import substrate_relations as sr
        a = np.array(Image.open(raw_path).convert("RGB"))
        if a.shape[:2] != (64, 64):
            s = max(1, a.shape[0] // 64)
            a = a[s // 2::s, s // 2::s][:64, :64]
        cols, cnts = np.unique(a.reshape(-1, 3), axis=0, return_counts=True)
        bg = cols[cnts.argmax()]                       # frame background = most common colour
        comps, colour = [], {}
        for e in (entities or []):
            bb = e.get("bbox_ticks_turn1")
            if not bb:
                continue
            r0, c0, r1, c1 = [int(x) for x in bb]
            crop = a[r0:r1 + 1, c0:c1 + 1]
            if crop.size == 0:
                continue
            mask = np.any(crop != bg, axis=2)
            if mask.sum() < 4:
                continue
            comps.append({"id": e["name"], "bbox": [r0, c0, r1, c1],
                          "mask": mask, "npix": int(mask.sum())})
            fg = crop[mask]
            fc, fn = np.unique(fg, axis=0, return_counts=True)
            colour[e["name"]] = tuple(int(x) for x in fc[fn.argmax()])
        rel = sr.find_relations(comps)
        out, seen = [], set()
        # SWAP pieces are the same shape at the SAME scale -> use exact same-mask /
        # same-canonical-shape groups only.  Deliberately NOT shape_similar (the graded
        # SCALE-invariant branch): a thin border edge can resemble a small node at a
        # different scale, which is not a swap pair.
        groups = list(rel.get("identical", [])) + list(rel.get("same_shape", []))
        pairs = [(g[i], g[j]) for g in groups
                 for i in range(len(g)) for j in range(i + 1, len(g))]
        for a_, b_ in pairs:
            if colour.get(a_) != colour.get(b_):          # same shape, DIFFERENT colour
                key = frozenset((a_, b_))
                if key not in seen:
                    seen.add(key)
                    out.append((a_, b_))
        return out
    except Exception:
        return []


def _normalize_vlm_rels(rels, entities):
    """The VLM AUTHORS the pairs (any type -- ball<->shell, key<->lock, swap, match,
    ...); normalize its free-form relationships to {from,to,relation,evidence}, keeping
    only pairs whose BOTH members are real entities.  (Pair recognition is the VLM's
    job -- there are too many pair types for fixed substrate rules; the substrate only
    VERIFIES geometry as evidence.)"""
    names = {e.get("name") for e in (entities or [])}
    out, seen = [], set()
    for r in (rels or []):
        if not isinstance(r, dict):
            continue
        bw = r.get("between")
        if isinstance(bw, (list, tuple)) and len(bw) >= 2:
            a, b = bw[0], bw[1]
        else:
            a, b = r.get("from"), r.get("to")
        rel = str(r.get("relation") or "related").strip()
        ev = str(r.get("note") or r.get("evidence") or "").strip()
        if a in names and b in names and a != b:
            key = tuple(sorted((str(a), str(b)))) + (rel,)
            if key not in seen:
                seen.add(key)
                out.append({"from": a, "to": b, "relation": rel, "evidence": ev})
    return out


ROLE_SYS = (
    "You are the PERCEPTION engine of a game-playing system. The substrate has ALREADY "
    "MEASURED every distinct object in the frame -- exact positions, sizes, colours, and "
    "which objects are identical or enclose one another. Your ONLY job is to look at the "
    "grid-annotated frame and assign each LISTED object a ROLE and a 2-3 word appearance. "
    "Do NOT add, remove, move, split, or merge objects -- the list is authoritative and "
    "complete. Roles (pick one per object): mover (a piece the player controls/moves), "
    "goal (a target/destination/slot a piece must reach or fill), marker (a small cue / "
    "centre dot / cursor on a piece), wall (an impassable barrier), terrain (floor / "
    "background structure), hud (a status bar / counter / border strip), other. Objects the "
    "substrate marks IDENTICAL are the SAME KIND; a tiny object ENCLOSED by a piece is a "
    "marker ON it. Reply ONE JSON object, first char `{`, no prose: "
    "{\"objects\":[{\"id\":\"obj0\",\"role\":<role>,\"appearance\":<2-3 words>}], "
    "\"game_type\":<short>, \"game_purpose\":<one sentence: what the player must do to win>}")

_ROLE_BASE = {"mover": "mover", "goal": "goal", "marker": "marker", "wall": "wall",
              "terrain": "terrain", "hud": "hud", "other": "object"}


def handle_perception(reply_path, prompt, is_delta):
    """Substrate-led perception: the substrate provides the COMPLETE measured object set
    (extract_objects -- figure-ground + fragment grouping + texture suppression) and the
    VLM only assigns a ROLE to each. A small VLM names kinds well but cannot localize on a
    64px grid, so this plays to each side's strength and removes the omission/duplication
    that the old VLM-led path fought with recovery machinery. Falls back to the VLM-led
    path if the substrate set is unavailable.

    DEFAULT is VLM-LED: identifying objects is intelligence (grouping fragments, ignoring
    textures, separating objects) the substrate LACKS -- a capable VLM should lead, with
    the substrate only GROUNDING its bboxes. Substrate-led is a CRUTCH for a VLM too weak
    to localize (e.g. Qwen3-VL-8B on a 64px grid); opt in with COS_SUBSTRATE_LED_PERCEPTION=1."""
    if not os.environ.get("COS_SUBSTRATE_LED_PERCEPTION"):
        return _handle_perception_vlm_led(reply_path, prompt, is_delta)
    raw = _latest_raw()
    img = reply_path.parent / "image_grid.png"
    if not img.exists():
        img = reply_path.parent / "curr_frame.png"
    objects = []
    try:
        import numpy as _np
        from PIL import Image as _Image
        import substrate_objects as _so
        import substrate_relations as _sr
        if raw:
            rgb = _np.array(_Image.open(raw).convert("RGB"))
            if rgb.shape[:2] == (64, 64):
                objects = _so.extract_objects(rgb)
    except Exception as e:
        print(f"[perception] substrate object-set unavailable ({e}); VLM-led fallback", flush=True)
    if not objects:
        return _handle_perception_vlm_led(reply_path, prompt, is_delta)

    rel = _sr.find_relations(objects)
    def _ctr(o):
        return o["center"]
    listing = "\n".join(
        f"  {o['id']}: center(row{_ctr(o)[0]},col{_ctr(o)[1]}), {o['npix']}px, colour {o['color']}"
        for o in objects)
    rel_lines = []
    for grp in rel.get("identical", []):
        rel_lines.append(f"  IDENTICAL: {', '.join(grp)}")
    for c in rel.get("contains", []):
        rel_lines.append(f"  {c['outer']} ENCLOSES {c['inner']}"
                         + (" (a tiny mark)" if c.get("inner_tiny") else ""))
    user = ("The substrate measured these objects (authoritative -- assign a role to EACH):\n"
            + listing + ("\n\nMeasured relations:\n" + "\n".join(rel_lines) if rel_lines else "")
            + "\n\nLook at the grid-annotated frame and output the JSON role assignment now:")
    ass = {}
    gtype = gpurpose = ""
    try:
        rep = _json_obj(_qwen(ROLE_SYS, user, str(img) if img.exists() else None, tier="deep"))
        for a in (rep.get("objects") or []) if rep else []:
            if isinstance(a, dict) and a.get("id"):
                ass[str(a["id"])] = a
        gtype = (rep or {}).get("game_type", "") or ""
        gpurpose = (rep or {}).get("game_purpose", "") or ""
    except Exception:
        pass

    entities, used = [], {}
    for o in objects:
        a = ass.get(o["id"], {})
        role = str(a.get("role") or "other").strip().lower()
        base = _ROLE_BASE.get(role, "object")
        used[base] = used.get(base, 0) + 1
        # singletons of the salient roles keep a bare name; everything else is numbered
        name = base if (used[base] == 1 and base in ("mover", "goal")) else f"{base}_{used[base]}"
        r0, c0, r1, c1 = o["bbox"]
        entities.append({
            "name": name, "bbox_ticks_turn1": [r0, c0, r1, c1],
            "center_ticks": [o["center"][0], o["center"][1]],
            "role_hypothesis": str(a.get("appearance") or role),
            "confidence": "medium", "color": o["color"]})

    relationships = []
    name_by_id = {o["id"]: e["name"] for o, e in zip(objects, entities)}
    for grp in rel.get("identical", []):
        nms = [name_by_id.get(i) for i in grp if name_by_id.get(i)]
        if len(nms) >= 2:
            relationships.append({"between": nms, "relation": "identical"})
    for c in rel.get("contains", []):
        a_, b_ = name_by_id.get(c["outer"]), name_by_id.get(c["inner"])
        if a_ and b_:
            relationships.append({"between": [a_, b_], "relation": "encloses"})

    perc = {"entities": entities, "groups": [], "relationships": relationships,
            "grid_inference": {"is_grid_based": False}, "symbolic_state": {},
            "game_type": gtype or _last_perception["game_type"],
            "game_purpose": gpurpose or _last_perception["game_purpose"],
            "overall_notes": "perception: substrate-measured object set; VLM-assigned roles"}
    _last_perception.update({"entities": entities, "game_type": perc["game_type"],
                             "game_purpose": perc["game_purpose"],
                             "relationships": relationships,        # so re-emit/grounding keep the pairs
                             "groups": perc.get("groups") or []})
    out = {"delta": {"agent_moved": True, "inferred_action": "CLICK",
                     "entities_changed": [], "summary": "see perception"},
           "perception": perc} if is_delta else perc
    reply_path.write_text(json.dumps(out), encoding="utf-8")
    return f"perception(substrate-led): {len(entities)} entities, type={perc['game_type'][:24]}"


def _covered_blob(b, entities):
    """A measured component is represented when an entity of the SAME measured colour
    overlaps it (colour-keyed so a small piece inside a larger region's bbox is not
    falsely 'covered')."""
    br0, bc0, br1, bc1 = b["bbox"]
    for e in entities:
        if (e.get("color") or "") != (b.get("color") or ""):
            continue
        r0, c0, r1, c1 = e["bbox_ticks_turn1"]
        if (max(0, min(r1, br1) - max(r0, br0))
                * max(0, min(c1, bc1) - max(c0, bc0))) > 0:
            return True
    return False


def _draw_review_overlay(raw_path, entities, out_path, up=10):
    """Substrate DRAWS the VLM's bboxes back onto the frame (clean NEAREST upscale +
    light tick grid + labelled red boxes) so the VLM can VISUALLY review and correct
    them (durable principle P17, step 2).  Returns the overlay path, or None on
    failure.  The substrate only draws + measures; the VLM decides the objects."""
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        a = np.array(Image.open(raw_path).convert("RGB"))
        s = max(1, a.shape[0] // 64)
        g = a[s // 2::s, s // 2::s][:64, :64]
        im = Image.fromarray(g, "RGB").resize((64 * up, 64 * up), Image.NEAREST).convert("RGB")
        d = ImageDraw.Draw(im)
        try:
            f = ImageFont.truetype("arial.ttf", 11)
        except Exception:
            f = ImageFont.load_default()
        for k in range(0, 65, 4):
            col = (110, 110, 110) if k % 16 else (90, 90, 0)
            d.line([(k * up, 0), (k * up, 64 * up)], fill=col)
            d.line([(0, k * up), (64 * up, k * up)], fill=col)
            if k % 16 == 0:
                d.text((k * up + 1, 0), str(k), fill=(255, 255, 0), font=f)
                d.text((0, k * up + 1), str(k), fill=(255, 255, 0), font=f)
        for e in entities:
            r0, c0, r1, c1 = e["bbox_ticks_turn1"]
            d.rectangle([c0 * up, r0 * up, (c1 + 1) * up, (r1 + 1) * up], outline=(255, 0, 0))
            d.text((c0 * up + 1, r0 * up + 1), str(e.get("name", "")), fill=(255, 0, 0), font=f)
        im.save(out_path)
        return out_path
    except Exception:
        return None


def _handle_perception_vlm_led(reply_path, prompt, is_delta):
    raw = _latest_raw()
    cands = _blobs(raw) if raw else []          # substrate MEASUREMENTS = unreliable candidates
    # ANCHOR-RELEASE: the driver sets PERCEPTION-RESET in the prompt when the PRIOR perception
    # grounded at QA~0 (its boxes were too COARSE -- the QA=0 lock).  Clear the stable-id tracker
    # so this turn re-identifies FINELY instead of inheriting the frozen coarse labels.  The
    # authoritative coarse signal is the grounding QA (fine-component), computed driver-side.
    if "PERCEPTION-RESET" in (prompt or ""):
        _component_tracker["next_id"] = 0
        _component_tracker["items"] = []
        _last_perception.pop("entities", None)
        print("[perception] PERCEPTION-RESET honored -> cleared stable-id anchor; re-perceiving "
              "finely", flush=True)
    # grid-annotated frame: axis tick labels let the VLM read coordinates itself
    img = reply_path.parent / "image_grid.png"
    if not img.exists():
        img = reply_path.parent / "curr_frame.png"
    listing = "\n".join(
        f"  candidate {i}: colour {b['color']}, bbox(row0,col0,row1,col1) {b['bbox']}, "
        f"{b['cells']} px" for i, b in enumerate(cands)) or "  (none measured)"
    rels = _relation_facts(raw) if raw else ""
    rel_block = (("\n\nMEASURED GEOMETRIC RELATIONS (AUTHORITATIVE -- exact pixel facts: "
                  "which shapes exist, which are IDENTICAL, which ENCLOSE others; trust "
                  "these for STRUCTURE, then assign meaning/roles yourself):\n" + rels)
                 if rels else "")
    # VLM-FIRST identification (durable principle P17): the VLM identifies the OBJECTS
    # from the IMAGE -- it is NOT seeded with a substrate object SET (objecthood stays the
    # VLM's judgement).  But the substrate DOES provide its measured non-background regions
    # as an ACCOUNTABILITY CHECKLIST: the VLM must cover every real region (merging the
    # parts of one object, omitting only stray specks).  Measured this raises coverage of
    # the substrate components from ~0-53% to ~75% AND anchors the VLM's tick coordinates
    # to exact pixel facts -- the substrate measures, the VLM still judges what is an object.
    # SUBSTRATE-TRACKED STABLE IDS + accountability + temporal anchoring, unified: the
    # substrate gives each measured region a STABLE id tracked across turns (matched by
    # colour + nearest position) AND the label the VLM gave that physical region last turn.
    # The VLM must account for every region and REUSE the shown label -- so identity is
    # robust THROUGH scene change (a vanished region drops, a new one gets a fresh id),
    # without relying on the VLM to keep names stable by instruction.
    if not is_delta:                              # level start -> fresh tracking
        _component_tracker["next_id"] = 0
        _component_tracker["items"] = []
    tracked = _track_components(cands)

    def _cl(t):                                   # neutral per-region listing (colour is reported as a
        lab = f" (last labelled '{t['label']}')" if t.get("label") else ""   # measurement, never a key)
        return f"C{t['id']} bbox {t['bbox']} colour {t['color']}{lab}"
    checklist = "; ".join(_cl(t) for t in tracked)
    accountability = (("\n\nThe substrate MEASURED these non-background regions, each with a STABLE id "
                       "(C#) tracked across turns -- ACCOUNT FOR EVERY ONE: cover it inside an object "
                       "(MERGE the regions that are parts of one object into that object's bbox), or "
                       "omit ONLY a truly isolated 1-2px speck. A region that sits ON, BETWEEN, or right "
                       "NEXT TO larger objects -- ESPECIALLY when several such regions RECUR -- is often "
                       "a STRUCTURAL MARKER (it can encode how the objects GROUP or PAIR, regardless of "
                       "its colour or shape): box EACH as its OWN SEPARATE entity, do NOT merge it into "
                       "an object it is attached to, and NEVER omit it as noise. For a region shown as "
                       "(last labelled 'X'), REUSE that exact name X for the object containing it -- "
                       "keeping names STABLE across turns is REQUIRED (downstream goals reference them); "
                       f"invent a name only for a region with no prior label. Measured regions ({len(cands)}): "
                       + checklist) if cands else "")
    # COORD-ADOPTION (weak-localizer assist, e.g. Qwen): a weak localizer recognizes object KINDS
    # but CONFABULATES coordinates. KEY: this clause is ADDITIVE inside the VLM's own identify-the-
    # objects frame -- it does NOT replace it with "just group the substrate's components" (that
    # variant is brittle: it collapsed to 0 on over-segmented scenes, e.g. bp35 with 103 substrate
    # components). A/B on OpenRouter Qwen3-VL-32B (cos_responder prompt vs +adopt vs group-only):
    # +adopt BEAT the gemma-tuned prompt at EVERY region count tested -- ka59(11) 59->63, r11l(13)
    # 62->75, sk48(26) 32->89, bp35(103) 33->29(noise). So the clause is safe across counts; the
    # ceiling is just conservatism on UNTESTED extreme over-segmentation (>30), where the measured
    # list is noise anyway. No-op for a strong localizer (its coords already match). Disablable via
    # COS_COORD_ADOPT_ASSIST=0.
    if (cands and len(cands) <= 30
            and os.environ.get("COS_COORD_ADOPT_ASSIST", "1") not in ("0", "false", "False")):
        accountability += ("\nThese measured bboxes are EXACT pixel facts: for each object you keep, "
                           "SET its bbox to the measured coordinates of the region(s) it covers -- "
                           "adopt those numbers VERBATIM, do NOT invent or eyeball coordinates. Your "
                           "job is to decide WHICH regions form one object and WHAT each is (role); "
                           "the coordinates are already measured for you.")
    user = ("Look at the grid-annotated frame (TICK coords; row 0 = top, col 0 = left) and identify "
            "EVERY distinct game object you SEE -- the player/mover, every piece, goals, "
            "markers/cursors, walls/barriers, LINES and wires, and small marks. A piece of a "
            "different colour from its surround is its OWN object; include thin lines and tiny "
            "marks; do not miss small ones. Give each a ROUGH bbox [row0,col0,row1,col1] read off "
            "the grid -- the substrate will measure it EXACTLY in the next step, so approximate is "
            "fine." + accountability + "\n\nOutput the JSON now:")
    # k=2: keep the RICHER identification (more real objects seen)
    obj = None
    for _ in range(2):
        cand = _json_obj(_qwen(PERC_SYS, user, str(img) if img.exists() else None, tier="deep"))
        if cand and cand.get("entities"):
            if obj is None or len(cand["entities"]) > len(obj.get("entities", [])):
                obj = cand
    obj = obj or {}
    used = {}
    entities = []

    # PIXEL->TICK REMAP (Qwen-class localizers): a weak localizer reports bboxes in the grid-annotated
    # render's PIXEL space (0..render_width, with the playfield INSET by the axis-label margin) and
    # IGNORES the tick-coord instruction -- so the *_ticks fields hold PIXELS and EVERY downstream tick
    # op silently fails (grounding containment 0/N, colour overlap empty, click delivery clamped to the
    # corner). The prompt-level coord-adopt clause above can't fix it (the model won't follow it). Detect
    # it (any VLM coord exceeds the tick grid) and remap by ALIGNING the VLM's overall bbox EXTENT to the
    # substrate's measured-component extent -- this recovers BOTH the scale AND the label-margin offset
    # with no render geometry, self-calibrating per frame. No-op for a strong localizer (coords already
    # tick-space) and when there are too few anchors to trust. Disablable via COS_PIXEL_TICK_REMAP=0.
    _NT = 64
    _remap = None
    if os.environ.get("COS_PIXEL_TICK_REMAP", "1") not in ("0", "false", "False"):
        try:
            _vb = [e.get("bbox") for e in (obj.get("entities") or [])
                   if isinstance(e.get("bbox"), (list, tuple)) and len(e.get("bbox")) == 4]
            _flat = [float(v) for b in _vb for v in b]
            # arm only when coords are CLEARLY render-pixel (>> the tick grid), so a strong
            # localizer that emits a slightly-out tick value (e.g. 65) is never wrongly remapped.
            if _flat and max(_flat) > 2 * _NT and len(_vb) >= 2 and len(cands) >= 2:
                vr0 = min(float(b[0]) for b in _vb); vc0 = min(float(b[1]) for b in _vb)
                vr1 = max(float(b[2]) for b in _vb); vc1 = max(float(b[3]) for b in _vb)
                mr0 = min(b["bbox"][0] for b in cands); mc0 = min(b["bbox"][1] for b in cands)
                mr1 = max(b["bbox"][2] for b in cands); mc1 = max(b["bbox"][3] for b in cands)
                sr = (mr1 - mr0) / ((vr1 - vr0) or 1.0); sc = (mc1 - mc0) / ((vc1 - vc0) or 1.0)

                def _remap(box):
                    r0, c0, r1, c1 = (float(v) for v in box)
                    return [mr0 + (r0 - vr0) * sr, mc0 + (c0 - vc0) * sc,
                            mr0 + (r1 - vr0) * sr, mc0 + (c1 - vc0) * sc]
                print(f"[perception] pixel->tick remap ARMED (VLM reports pixels): extent "
                      f"rows[{vr0:.0f}..{vr1:.0f}] cols[{vc0:.0f}..{vc1:.0f}] -> measured "
                      f"rows[{mr0}..{mr1}] cols[{mc0}..{mc1}], scale {sr:.3f},{sc:.3f}", flush=True)
        except Exception:
            _remap = None

    def _emit(e, dedup=False):
        bb = e.get("bbox") if isinstance(e, dict) else None
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            return
        try:
            r0, c0, r1, c1 = (float(x) for x in bb)
        except Exception:
            return
        if _remap is not None:                       # render-pixel bbox -> tick grid (see above)
            r0, c0, r1, c1 = _remap((r0, c0, r1, c1))
        r0, c0, r1, c1 = (int(round(v)) for v in (r0, c0, r1, c1))
        if r1 < r0:
            r0, r1 = r1, r0
        if c1 < c0:
            c0, c1 = c1, c0
        if dedup:                                    # skip an object already listed
            for ex in entities:
                xr0, xc0, xr1, xc1 = ex["bbox_ticks_turn1"]
                if (max(0, min(r1, xr1) - max(r0, xr0))
                        * max(0, min(c1, xc1) - max(c0, xc0))) > 0:
                    return
        role = (e.get("role") or "object").strip().lower()
        base = {"mover": "mover", "goal": "goal", "marker": "marker", "hud": "hud",
                "wall": "wall", "decoration": "decor"}.get(role, "object")
        used[base] = used.get(base, 0) + 1
        name = (str(e.get("name") or "").strip()
                or (base if used[base] == 1 and base in ("mover", "goal") else f"{base}_{used[base]}"))
        # measured COLOUR from the best-overlapping substrate candidate (geometry stays
        # substrate-owned; the driver's entity_grounding snaps the bbox precisely).
        col, best = "", 0
        for b in cands:
            br0, bc0, br1, bc1 = b["bbox"]
            ov = max(0, min(r1, br1) - max(r0, br0)) * max(0, min(c1, bc1) - max(c0, bc0))
            if ov > best:
                best, col = ov, b["color"]
        entities.append({
            "name": name, "bbox_ticks_turn1": [r0, c0, r1, c1],
            "center_ticks": [round((r0 + r1) / 2), round((c0 + c1) / 2)],
            "role_hypothesis": str(e.get("appearance") or role),
            "confidence": "medium", "color": col})

    for e in (obj.get("entities") or []):
        _emit(e)

    # COMPLETENESS ESCALATION (game-agnostic): the substrate measured distinct
    # components the VLM did NOT account for -> RE-ASK the VLM to identify ONLY
    # those, then merge.  The VLM still does the identifying (VLM-led); the
    # substrate only guarantees that no distinct object is silently dropped -- the
    # failure mode where a SECOND same-colour piece (e.g. a second green square)
    # gets deduped away, leaving the actual player invisible.  One extra round.
    def _covered(b):
        # A measured blob is represented only by an entity of the SAME measured
        # colour.  A small distinct-colour piece (e.g. a green square) is NOT
        # "covered" by the large grey region whose bbox merely ENCLOSES it -- that
        # bare-overlap bug is exactly what hid the player (green_white sits inside
        # left_platform's bbox).  Colour-keyed coverage forces it to be flagged.
        br0, bc0, br1, bc1 = b["bbox"]
        for e in entities:
            if (e.get("color") or "") != (b.get("color") or ""):
                continue
            r0, c0, r1, c1 = e["bbox_ticks_turn1"]
            if (max(0, min(r1, br1) - max(r0, br0))
                    * max(0, min(c1, bc1) - max(c0, bc0))) > 0:
                return True
        return False
    missed = [b for b in cands if not _covered(b)]
    if missed and img.exists():
        lst = "\n".join(
            f"  object {i}: colour {b['color']}, bbox(row0,col0,row1,col1) {b['bbox']}, "
            f"{b['cells']} px" for i, b in enumerate(missed))
        ru = ("Your perception OMITTED these measured objects -- the substrate found "
              "them but you did NOT list them. They are REAL distinct game objects "
              "(commonly a SECOND piece of the SAME colour you merged into one, or a "
              "small piece such as the player). Look at the frame and identify EACH:\n"
              + lst + "\n\nReturn JSON {\"entities\":[{name,bbox,role,appearance}]} for "
              "ONLY these missed objects now:")
        try:
            n0 = len(entities)
            add = _json_obj(_qwen(PERC_SYS, ru, str(img), tier="deep"))
            for e in (add.get("entities") or []) if add else []:
                _emit(e, dedup=True)
            if len(entities) > n0:
                print(f"[perception] completeness escalation: recovered "
                      f"{len(entities) - n0} of {len(missed)} missed object(s)",
                      flush=True)
        except Exception:
            pass

    # DROPPED SAME-COLOUR OBJECT RECOVERY: the VLM reliably labels ONE of two
    # similar same-colour pieces and silently drops the other, and cannot recover
    # it even when pointed at its exact bbox (the invisible-green-player failure).
    # So if a MISSED candidate has a COVERED, similar-sized, same-colour piece the
    # VLM DID label, recover the missed one as a SEPARATE object.  Crucially the
    # similar-size + same-colour test is only a DETECTOR of "the VLM probably
    # merged a piece away here" -- NOT a claim that the two are the same thing.
    # The two are presumed DISTINCT and are characterised independently (open
    # role, own identity; see below), because same-colour pieces routinely differ
    # in kind.  No tuned threshold -- similar-size + a covered same-colour piece
    # are structural -- and scoped narrowly to a failure the VLM provably cannot
    # fix.
    def _dims(bb):
        return (bb[2] - bb[0], bb[3] - bb[1])
    for b in cands:
        if _covered(b):
            continue
        bw, bh = _dims(b["bbox"])
        twin = next((c for c in cands if c is not b
                     and (c.get("color") or "") == (b.get("color") or "")
                     and abs(_dims(c["bbox"])[0] - bw) <= 1
                     and abs(_dims(c["bbox"])[1] - bh) <= 1 and _covered(c)), None)
        if twin is None:
            continue
        # SKIP a field/fill colour: if this colour ALSO paints a LARGER, non-
        # congruent region (a platform, the floor), then b is a same-colour PATCH
        # of that region (ground), not a distinct piece.  A true piece colour
        # (e.g. the green players) paints only small congruent blobs.  Structural
        # (strictly-larger + non-congruent), no tuned threshold, no bg-colour key.
        if any((c.get("color") or "") == (b.get("color") or "")
               and c["cells"] > b["cells"]
               and not (abs(_dims(c["bbox"])[0] - bw) <= 1
                        and abs(_dims(c["bbox"])[1] - bh) <= 1)
               for c in cands):
            continue
        # The dropped piece SHARES A COLOUR with a labelled entity but is a
        # DISTINCT object -- it is NOT a copy of that entity.  Two same-colour
        # pieces are routinely different in kind (one carries a mark -> player,
        # the other is a plain block), so there is no justification to assume
        # the recovered piece shares the sibling's role.  Inheriting the
        # sibling's role + kind-name is exactly what made the two greens
        # indistinguishable downstream, so the tracker locked the wrong one.
        # Recover it with its OWN OPEN identity: a neutral "object" name and
        # role, role left for the motion-based identifier / probe to determine
        # independently.  Record the same-colour sibling so the explore phase
        # knows these two are distinct-but-similar and must be DISAMBIGUATED.
        sib = next((e for e in entities
                    if (e.get("color") or "") == (b.get("color") or "")), None)
        used["object"] = used.get("object", 0) + 1
        bb = b["bbox"]
        entities.append({
            "name": f"object_{used['object']}", "bbox_ticks_turn1": [int(x) for x in bb],
            "center_ticks": [round((bb[0] + bb[2]) / 2), round((bb[1] + bb[3]) / 2)],
            "role_hypothesis": "object", "confidence": "low",
            "color": (b.get("color") or ""),
            "same_colour_as": (sib.get("name") if sib else None),
            "needs_disambiguation": True})
        print(f"[perception] dropped same-colour object recovered at {bb} "
              f"(colour {b.get('color')}) -- VLM labelled a same-colour piece but "
              f"merged this DISTINCT one away; role left OPEN for probing", flush=True)
    # ITERATIVE VISUAL-REVIEW LOOP (durable principle P17): the substrate DRAWS the
    # VLM's bboxes on the frame + lists exact measured coords; the VLM REVIEWS the
    # overlay against the image and corrects (fix bbox / merge / split / add missed /
    # drop phantom); loop until the VLM is satisfied.  Soft cap 3 passes; the VLM may
    # PUSH for more (need_more_passes) up to a hard cap of 6.  The substrate only draws
    # + measures here -- the VLM owns which objects exist.
    _SOFT, _HARD, _it = 3, 6, 0
    _vlm_rels = []                              # the VLM AUTHORS the pairs (any type)
    while entities and raw and img.exists() and _it < _HARD:
        overlay = _draw_review_overlay(raw, entities, reply_path.parent / "perc_review.png")
        if overlay is None:
            break
        coords = "\n".join(f"  {e['name']}: measured bbox(r0,c0,r1,c1) {e['bbox_ticks_turn1']}"
                           for e in entities)
        missed = [b for b in cands if not _covered_blob(b, entities)]
        miss = (("\nThe substrate ALSO measured these components you did NOT box (ADD any that are a "
                 "real object -- e.g. a line/wire or a second same-colour piece): "
                 + "; ".join(f"{b['color']}@{b['bbox']}" for b in missed)) if missed else "")
        pairs = _pair_relations(raw, entities) if raw else []
        pair_hint = (("\nThe substrate VERIFIED these geometric relations as EVIDENCE (you decide which "
                      "are meaningful pairs): "
                      + "; ".join(f"{p['relation']}({p['from']}<->{p['to']})" for p in pairs[:8])) if pairs else "")
        swap_cands = _shape_swap_candidates(raw, entities) if raw else []
        if swap_cands:
            print(f"[perception] same-shape/contrasting pair candidate(s): "
                  + "; ".join(f"{a}<->{b}" for a, b in swap_cands[:6]), flush=True)
        _many_pairs = len(swap_cands) >= 3
        swap_hint = (("\nSAME-SHAPE / CONTRASTING pairs -- the substrate measured these as the SAME "
                      "shape but a DIFFERENT colour/marking: "
                      + "; ".join(f"{a}<->{b}" for a, b in swap_cands[:8])
                      + ".\nDISAMBIGUATE 'swap' vs 'match' by STRUCTURE (do not guess, do not default):\n"
                        "- 'swap' = two MUTUALLY-MOVABLE peers EXCHANGE positions; it requires BOTH "
                        "sides to be editable/movable AND an action that visibly EXCHANGES them. If one "
                        "side is a FIXED reference (an example/legend region that does not change), it "
                        "CANNOT be swapped -- then it is 'match'.\n"
                        "- 'match' = make an EDITABLE element CONFORM to a REFERENCE element; a FIXED "
                        "example region paired with a separate EDITABLE region means APPLY the reference "
                        "to the target.\n"
                      + ("- These pairs are MANY and PARALLEL (a repeated table). A consistent lookup "
                         "TABLE / LEGEND to APPLY is far simpler and more likely than many independent "
                         "swaps -- prefer 'match' UNLESS you can actually see a single action that "
                         "EXCHANGES one pair.\n" if _many_pairs else "")
                      + "Emit the relation the STRUCTURE supports.")
                     if swap_cands else "")
        _it += 1
        ru = ("Attached: the frame with YOUR boxes DRAWN + labelled. The substrate's EXACT measured "
              "coordinates:\n" + coords + miss + pair_hint + swap_hint +
              "\n\nREVIEW your boxes against the image. Is every real object boxed, each box tight and "
              "on the correct object? FIX any bbox, MERGE boxes that are one object, SPLIT a box that "
              "covers two, ADD a missed object, DROP a phantom.\n"
              "Then IDENTIFY every RELATED PAIR you see and label each with the RELATION TYPE that fits "
              "(put ONE of these words in 'relation'):\n"
              "- 'connector': A and B joined by a LINE/BAR/BRIDGE = a defined association (often a LEGEND "
              "entry binding A->B).\n"
              "- 'fits' / 'contains': A fits into or is enclosed by B (slot, shell, ring, lock, container).\n"
              "- 'match': make an EDITABLE A conform to a REFERENCE B (apply a legend/template).\n"
              "- 'swap': two MUTUALLY-MOVABLE peers EXCHANGE positions.\n"
              "- 'mirror': A and B are mirror / symmetric twins.\n"
              "- 'reach': A is a controllable MOVER and B is its GOAL/target (NOT a swap).\n"
              "- 'gate': A is a conditional PASSAGE / PORTAL for B.\n"
              "- 'identical': same shape AND colour (duplicates).\n"
              "Disambiguate same-shape/contrasting by STRUCTURE (above): 'swap' ONLY for mutually-movable "
              "peers an action EXCHANGES; 'match' when one side is a FIXED reference or many parallel pairs "
              "form a table/legend to APPLY; a mover and its goal are 'reach', not swap. "
              "Use the substrate's verified relations + the same-shape/contrasting pairs above as "
              "EVIDENCE, but rely on YOUR visual understanding to choose the pair, the relation, and what "
              "it MEANS (e.g. 'put the ball in the shell to win'). Return the FULL corrected "
              "list as {\"entities\":[{name,role,bbox,appearance}], "
              "\"relationships\":[{between:[a,b],relation,note}], \"satisfied\": <true iff nothing needs "
              "changing>, \"need_more_passes\": <true ONLY if you still need another review after this>}.")
        try:
            rev = _json_obj(_qwen(PERC_SYS, ru, str(overlay), tier="deep")) or {}
        except Exception:
            break
        if rev.get("relationships"):
            _vlm_rels = rev["relationships"]   # keep the latest pass's pairs (VLM-authored)
        new = [e for e in (rev.get("entities") or []) if isinstance(e, dict) and e.get("bbox")]
        if new:
            entities = []
            used = {}
            for e in new:
                _emit(e)
        sat, push = bool(rev.get("satisfied")), bool(rev.get("need_more_passes"))
        if sat and not push:
            print(f"[perception] review loop: VLM satisfied after {_it} pass(es), "
                  f"{len(entities)} entities", flush=True)
            break
        if _it >= _SOFT and not push:
            print(f"[perception] review loop: soft cap {_SOFT} reached, {len(entities)} entities",
                  flush=True)
            break
        if push and _it >= _SOFT:
            print(f"[perception] review loop: VLM pushed for pass {_it + 1} (> soft cap {_SOFT})",
                  flush=True)

    # RECOVER structural sub-markers the VLM MERGED.  GAME-AGNOSTIC, keyed only on RELATIONAL
    # signals -- never on colour or shape (those carry no fixed meaning across games):
    #   1. ADJACENCY -- the component sits ON / BETWEEN / right NEXT TO a recognized entity
    #      (its centre is within a couple ticks of some entity's bbox), so it is attached to
    #      something that matters, not a stray speck floating in empty space.
    #   2. REPETITION -- two or more such attached components recur (a regular pattern is
    #      strong structural evidence, e.g. tr87's gray pairing-bars / tile value-marks).
    #   3. NOT-THE-OBJECT -- it is not itself a boxed object: within each entity the LARGEST
    #      measured component inside it IS that object (threshold-free); the rest are markers.
    # Measurement, not meaning -- the role is a low-confidence guess for the VLM/downstream.
    def _area(_t):
        _b = _t["bbox"]
        return (_b[2] - _b[0] + 1) * (_b[3] - _b[1] + 1)

    def _adjacent(_cb, _eb):                      # bboxes OVERLAP or TOUCH (neighbouring cells) --
        return (_cb[0] <= _eb[2] + 1 and _eb[0] <= _cb[2] + 1   # geometric adjacency, no tuned margin
                and _cb[1] <= _eb[3] + 1 and _eb[1] <= _cb[3] + 1)
    _ent_bb = [e["bbox_ticks_turn1"] for e in entities
               if isinstance(e.get("bbox_ticks_turn1"), (list, tuple)) and len(e["bbox_ticks_turn1"]) == 4]
    _represented = set()                          # the LARGEST component in an entity IS that object
    for _eb in _ent_bb:
        _inside = [t for t in tracked if _eb[0] <= t["center"][0] <= _eb[2]
                   and _eb[1] <= t["center"][1] <= _eb[3]]
        if _inside:
            _represented.add(id(max(_inside, key=_area)))
    def _inside_any(_ct):                         # centre falls inside some entity's bbox
        return any(_eb[0] <= _ct[0] <= _eb[2] and _eb[1] <= _ct[1] <= _eb[3] for _eb in _ent_bb)
    # Recover only CONNECTORS -- markers that sit BETWEEN/BESIDE objects (touching an entity but
    # whose centre is NOT inside any entity), e.g. the bars linking tiles.  A marker whose centre
    # is INSIDE an object is that object's own content (a glyph/value already carried by the
    # object's crop) -- recovering it as a separate entity floods the inventory and creates
    # overlap-conflicts that degrade grounding.  Between-vs-inside is structural, not colour/shape.
    _attached = [t for t in tracked if id(t) not in _represented and not _inside_any(t["center"])
                 and any(_adjacent(t["bbox"], _eb) for _eb in _ent_bb)]
    if len(_attached) >= 2:                       # REPETITION + ADJACENCY (between objects) -> connectors
        for _t in _attached:
            _bb, _ct = _t["bbox"], _t["center"]
            used["marker"] = used.get("marker", 0) + 1
            entities.append({"name": f"marker_{used['marker']}",
                             "bbox_ticks_turn1": [int(x) for x in _bb],
                             "center_ticks": [round(_ct[0]), round(_ct[1])],
                             "role_hypothesis": "recurring marker adjacent to objects (grouping/structure clue)",
                             "confidence": "low", "color": _t["color"], "needs_disambiguation": True})
            print(f"[perception] recovered marker at {_bb} (adjacent to an entity, {len(_attached)} "
                  f"recur) -- VLM merged it; structural clue (colour/shape not used)", flush=True)

    # Pairs are VLM-AUTHORED (it identifies compatible/complementary pairs of any type);
    # the substrate is only a fallback/assist when the VLM names none (P17).
    relationships = _normalize_vlm_rels(_vlm_rels, entities)
    _src = "VLM"
    if not relationships and raw:
        relationships = _pair_relations(raw, entities)
        _src = "substrate-fallback"
    if relationships:
        print(f"[perception] {len(relationships)} compatible/complementary pair(s) [{_src}]: "
              + "; ".join(f"{r['relation']}({r['from']}<->{r['to']})" for r in relationships[:6]), flush=True)
    perc = {"entities": entities, "groups": [], "relationships": relationships,
            "grid_inference": {"is_grid_based": False}, "symbolic_state": {},
            "game_type": obj.get("game_type", "") or _last_perception["game_type"],
            "game_purpose": obj.get("game_purpose", "") or _last_perception["game_purpose"],
            "overall_notes": "perception: VLM-driven identification, substrate-assisted "
                             "iterative review (P17); relationships = measured "
                             "compatible/complementary pairs"}
    _label_components(entities)                 # carry each region's name to the next frame (stable ids)
    # ANCHOR-RELEASE on a COARSE perception: if the VLM boxed FAR FEWER entities than the
    # substrate measured components (under-segmented -- the QA=0 coarse-lock seen on tr87, where
    # 16 super-boxes spanned 63 components), do NOT make this the anchor.  Keep the prior (finer)
    # inventory so the NEXT turn anchors to it and re-perceives finely, instead of temporal
    # anchoring freezing the coarse view.  Threshold is derived (boxing < half the measured
    # components = clear under-coverage), not tuned to a game; only fires on a delta with a
    # finer prior to fall back to.
    _coarse = (bool(raw) and is_delta and (_last_perception.get("entities"))
               and len(cands) > 8 and len(cands) >= 2 * max(1, len(entities)))
    if _coarse:
        print(f"[perception] COARSE view ({len(entities)} entities vs {len(cands)} measured "
              f"components) -> NOT anchoring; keeping the prior finer inventory", flush=True)
    else:
        _last_perception.update({"entities": entities, "game_type": perc["game_type"],
                                 "game_purpose": perc["game_purpose"],
                                 "relationships": relationships,    # so re-emit/grounding keep the pairs
                                 "groups": perc.get("groups") or []})
    out = {"delta": {"agent_moved": True, "inferred_action": "CLICK",
                     "entities_changed": [], "summary": "see perception"},
           "perception": perc} if is_delta else perc
    reply_path.write_text(json.dumps(out), encoding="utf-8")
    return f"perception: {len(entities)} entities, type={perc['game_type'][:24]}"


MENU_SYS = (
    "You are PLAYING a colour-merge puzzle (2048-style): merging two pieces of the "
    "SAME colour advances them up a legend chain (cyan->magenta->purple->yellow...); "
    "when only ONE finished piece is left, deliver it onto the goal blob.  The "
    "substrate gives you a MENU of candidate operators (each is one click) and the "
    "attached scene.  Choose the single best operator for THIS turn.  Reply with ONE "
    "JSON object only: {\"choice\": <integer index>}, no prose.")


def handle_merge_menu(reply_path, prompt):
    menu = prompt[prompt.index("MERGE OPERATOR MENU"):]
    ops = re.findall(r"-\s*(.+?)\s*->\s*(CLICK:\d+,\d+)", menu)
    mrec = re.search(r"recommendation:\s*(CLICK:\d+,\d+)", menu)
    rec = mrec.group(1) if mrec else (ops[0][1] if ops else "NONE")
    # Small-model compensation: the SUBSTRATE owns the merge ordering -- it measures the
    # same-colour pairs and recommends the CLOSEST (minimising walk distance), a geometry/
    # sequencing task the model is unreliable at (and which the menu is usually a single
    # operator for anyway).  Trust the recommendation instead of asking the VLM to pick an
    # index; that keeps the merge chain on the substrate's plan and saves a VLM call.
    act = rec
    reply = {"endorsed_action": act,
             "rationale": "substrate-recommended merge operator (closest same-colour pair)",
             "confidence": "high", "game_type": _last_perception["game_type"],
             "game_purpose": _last_perception["game_purpose"]}
    reply_path.write_text(json.dumps(reply), encoding="utf-8")
    return f"merge-menu: substrate-rec {act} of {len(ops)}"


def handle_strategy(reply_path, prompt):
    # OPERATOR MENU: the substrate enumerated merge/deliver operators -> the VLM SELECTS.
    if "MERGE OPERATOR MENU" in prompt:
        return handle_merge_menu(reply_path, prompt)
    # ENDORSE the mechanical actor's PROPOSED action.  Read it from the prompt's
    # CANONICAL statement of the planner's choice -- "planner chose '<X>'" / a line
    # "action: <X>" -- NOT the first ACTION token in the instructional prose.  The old
    # regex grabbed a stray ACTION1 from the help text and endorsed it EVERY turn,
    # overriding the planner and freezing the run on a cardinal the game ignores.
    proposed = None
    for pat in (r"planner chose '([^']+)'",
                r"(?im)^\s*action:\s*(CLICK:[0-9]+,[0-9]+|CLICK:[A-Za-z0-9_./-]+|ACTION[0-9]|UP|DOWN|LEFT|RIGHT|NONE)",
                r"proposed[^\n]*?\b(ACTION[0-9]|CLICK:[0-9]+,[0-9]+)"):
        m = re.search(pat, prompt, re.I)
        if m:
            proposed = m.group(1).strip()
            break
    proposed = proposed or "NONE"
    reply = {"endorsed_action": proposed,
             "rationale": "Endorse the substrate's proposed action; the substrate owns "
                          "the move-law / decomposition (small-model compensation).",
             "confidence": "medium",
             "game_type": _last_perception["game_type"],
             "game_purpose": _last_perception["game_purpose"]}
    reply_path.write_text(json.dumps(reply), encoding="utf-8")
    return f"strategy: endorse {proposed}"


ANIM_SYS = (
    "You read a short game ANIMATION filmstrip (sub-frames left->right, earliest "
    "to latest) showing what happened right after one action. Reading motion is "
    "your strength -- be precise about WHICH object moved and FROM where TO where. "
    "Reply with ONE JSON object only, first char `{`, last char `}`, no prose.\n"
    "Schema: {\"animation_analysis\": {\"scene_summary\": <1 sentence>, "
    "\"movements\": [<one string per moved object: '<colour/role> moved from "
    "(row,col) to (row,col)' or 'grew/shrank/recoloured ...'>], "
    "\"win_understood\": <one sentence: what the player must do to WIN, inferred "
    "from what this animation DEMONSTRATES>}}")


def handle_animation(reply_path, prompt):
    img = reply_path.parent / "animation_filmstrip.png"
    blobs = _blobs(_latest_raw()) if _latest_raw() else []
    hint = "; ".join(f"{b['color']}@({b['center'][0]},{b['center'][1]})" for b in blobs[:6])
    user = ("The filmstrip shows the sub-frames (earliest LEFT -> latest RIGHT) of "
            "what happened after the last action. Current objects: " + hint + ".\n"
            "Describe precisely which object MOVED and from where to where, and "
            "what the action demonstrates about how to win. Output the JSON now:")
    obj = _json_obj(_qwen(ANIM_SYS, user, str(img) if img.exists() else None, max_tokens=1500, tier="deep"))
    if not (obj and obj.get("animation_analysis")):
        obj = {"animation_analysis": {"scene_summary": "an object moved after the action",
               "movements": [], "win_understood": _last_perception["game_purpose"] or "move the controlled piece onto the goal"}}
    aa = obj["animation_analysis"]
    aa.setdefault("scene_summary", "an object moved")
    aa.setdefault("movements", [])
    aa.setdefault("win_understood", _last_perception["game_purpose"] or "reach the goal")
    reply_path.write_text(json.dumps(obj), encoding="utf-8")
    return f"animation: {len(aa['movements'])} moves; win={str(aa['win_understood'])[:46]}"


def handle_gate(reply_path, name):
    n = name.lower()
    if "refinement" in n or "requery" in n:          # re-emit cached perception (no re-derive)
        body = dict(_last_perception_perc())
    elif "completeness" in n:
        body = {"entities": []}                        # nothing to add
    elif "audit" in n:
        body = {"entities": _last_perception["entities"]}
    elif "inspection" in n:
        body = {"verdict": "accept", "entities_ok": True, "notes": "substrate-grounded"}
    elif "animation" in n:
        body = {"animation_analysis": {"win_understood": _last_perception["game_purpose"]}}
    else:
        body = {"entities": _last_perception["entities"]}
    reply_path.write_text(json.dumps(body), encoding="utf-8")
    return f"gate {n[:28]}"


EOT_SYS = (
    "You are reviewing a game trial that just ENDED.  Read the playback (the actions "
    "you took) and look at the attached END frame.  If the level was not completed, "
    "work out the ROOT CAUSE and a concrete CORRECTIVE strategy for next time.  Reply "
    "with ONE JSON object matching the schema in the prompt (the keys it lists), no "
    "prose, first char `{`, last char `}`.")


def handle_end_of_trial(reply_path, prompt):
    # the END frame so the VLM SEES the board it failed on
    frames = [p for p in glob.glob(str(WORK / "turn_*" / "curr_frame.png"))]
    img = None
    if frames:
        img = sorted(frames, key=lambda p: int(os.path.basename(os.path.dirname(p)).split("_")[1])
                     if os.path.basename(os.path.dirname(p)).split("_")[1].isdigit() else 0)[-1]
    obj = _json_obj(_qwen(EOT_SYS, prompt, img, max_tokens=1800)) or {}
    body = {"free_form_lessons": obj.get("free_form_lessons", []) or [],
            "refuted_alternatives": obj.get("refuted_alternatives", []) or [],
            "trial_notes": obj.get("trial_notes", "") or "",
            "game_summary": obj.get("game_summary", "") or "",
            "strategic_approach": obj.get("strategic_approach", "") or "",
            "failure_diagnosis": obj.get("failure_diagnosis", "") or "",
            "corrective_strategy": obj.get("corrective_strategy", "") or ""}
    reply_path.write_text(json.dumps(body), encoding="utf-8")
    return ("end-of-trial review: diagnosis=%r corrective=%r"
            % (body["failure_diagnosis"][:50], body["corrective_strategy"][:50]))


def _last_perception_perc():
    return {"entities": _last_perception["entities"],
            "groups": _last_perception.get("groups") or [],
            "relationships": _last_perception.get("relationships") or [],
            "grid_inference": {"is_grid_based": False}, "symbolic_state": {},
            "game_type": _last_perception["game_type"],
            "game_purpose": _last_perception["game_purpose"],
            "overall_notes": "re-emit (substrate-grounded)"}


# ---------------------------------------------------------------------------
# watch loop
# ---------------------------------------------------------------------------
def _reply_for(prompt_path, text=""):
    # Some prompts (mandatory instincts) name an EXPLICIT reply file that is NOT a
    # simple prompt->reply rename (e.g. instinct_animation_first_prompt.md asks for
    # `reply_animation_first_reply.txt`). Honour an explicit "Write the JSON to:".
    m = re.search(r"[Ww]rite (?:the|your) (?:JSON|reply)\b[\s\S]{0,40}?`([A-Za-z0-9_./-]+\.txt)`", text)
    if m:
        return prompt_path.with_name(Path(m.group(1).strip()).name)
    name = prompt_path.name.replace("prompt", "reply").replace(".md", ".txt")
    return prompt_path.with_name(name)


def main(stop_evt=None):
    print(f"[cos-responder] watching {WORK} model={MODEL}", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    answered = {}                       # reply_path -> prompt mtime we answered
    idle = 0
    while idle < 200 and not (stop_evt is not None and stop_evt.is_set()):
        pending = []
        for p in WORK.glob("**/*prompt*.md"):
            try:
                pm = p.stat().st_mtime
                ptext = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            r = _reply_for(p, ptext)
            # answer a prompt we haven't answered, OR a re-prompt (prompt rewritten
            # newer than our last answer -- e.g. a mandatory-instinct re-prompt)
            if answered.get(str(r), -1.0) >= pm:
                continue
            pending.append((p, r, pm, ptext))
        if not pending:
            idle += 1; time.sleep(2); continue
        idle = 0
        for p, r, pm, ptext in pending:
            nm = p.name
            t0 = time.time()
            try:
                if nm == "prompt.md":
                    is_delta = (p.parent / "prev_frame.png").exists() and "turn_001" not in str(p)
                    msg = handle_perception(r, ptext, is_delta)
                elif "strategy" in nm:
                    msg = handle_strategy(r, ptext)
                elif "animation" in nm:
                    msg = handle_animation(r, ptext)
                elif "end_of_trial" in nm:
                    msg = handle_end_of_trial(r, ptext)
                else:
                    msg = handle_gate(r, nm)
            except Exception as e:
                r.write_text(json.dumps({"entities": [], "note": f"responder error {e}"}), encoding="utf-8")
                msg = f"ERROR {str(e)[:80]}"
            answered[str(r)] = pm
            _sp = (f" | ${_cost['usd']:.4f}" if _cost.get("usd") else "")
            print(f"[{p.parent.name}/{nm}] {time.time()-t0:.1f}s -> {msg}{_sp}", flush=True)
    print(f"[cos-responder] idle-exit. cost={_cost}", flush=True)


def serve(work_dir, model_slug, game_id, raw_dir=None, stop_evt=None, deep_model=None):
    """In-process entry: point the responder at a driver's work_dir + a model slug,
    answering its file-handoff prompts via backends.call_oracle (the offline qwen
    path).  Run on a daemon thread alongside the ExploratoryDriver in the bridge.

    deep_model (optional): a stronger slug for the HARD visual steps (perceive /
    read-animation) -- gated CoT.  Defaults to model_slug (single-model), so passing
    nothing leaves behaviour unchanged.  VLM-agnostic: any backends.call_oracle slug."""
    global WORK, RAW_DIR, MODEL, FAST_MODEL, DEEP_MODEL, OLLAMA_URL
    WORK = Path(work_dir)
    RAW_DIR = Path(raw_dir) if raw_dir else (REPO / ".tmp/exploratory_play_frames" / game_id)
    MODEL = model_slug
    FAST_MODEL = model_slug
    DEEP_MODEL = deep_model or os.environ.get("COS_DEEP_VLM_MODEL") or model_slug
    OLLAMA_URL = None    # route via backends.call_oracle(slug) -> call_ollama (num_gpu=99)
    # Record which model actually drives the run so the trace's Run-information
    # header shows it.  The driver seeds run_info.json with game/level but defers
    # the model to whoever calls it (this responder).  Merge, never clobber:
    # read -> set acting/perception model -> write, so the driver's baseline keys
    # survive regardless of write order.
    try:
        ri = WORK / "run_info.json"
        info = {}
        if ri.exists():
            info = json.loads(ri.read_text(encoding="utf-8")) or {}
        info["acting_model"] = MODEL
        info["acting_provider"] = "offline (backends.call_oracle -> ollama)"
        info["perception_model"] = DEEP_MODEL if DEEP_MODEL != MODEL else MODEL
        ri.write_text(json.dumps(info, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[run-info] responder could not record model: {e}")
    main(stop_evt=stop_evt)


if __name__ == "__main__":
    main()
