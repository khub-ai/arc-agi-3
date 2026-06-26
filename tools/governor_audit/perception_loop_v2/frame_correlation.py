"""frame_correlation.py -- per-frame co-occurrence: changes that happen in the SAME
frame of an animation sequence are likely RELATED.

This is the fine-grained complement to across-frames salient co-occurrence.  When one
frame of a sequence shows TWO (or more) regions changing together -- e.g. the mover
steps up one cell AND one specific switch column highlights -- those changes are a
designer's binding: emit a correlation claim tying them.  Game- and domain-agnostic
("frame" = any synchronized observation step; the same logic binds a robot gripper's
motion to the object that moves with it).

Pure; numpy only.  Returns measured FACTS (which regions changed together, when); the
caller maps regions to named entities and files the GUESSED relation claim.
"""
from __future__ import annotations

from collections import deque

import numpy as np


def _changed_mask(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return None
    return (a != b) if a.ndim == 2 else np.any(a != b, axis=-1)


def _components(mask, min_px: int):
    """4-connectivity connected components of a boolean mask -> [{bbox,npix,center}]."""
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(H):
        row = mask[r]
        for c in range(W):
            if not row[c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            ys, xs = [], []
            while q:
                y, x = q.popleft()
                ys.append(y); xs.append(x)
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if len(ys) >= min_px:
                out.append({"bbox": [min(ys), min(xs), max(ys), max(xs)], "npix": len(ys),
                            "center": [round(sum(xs) / len(xs), 1), round(sum(ys) / len(ys), 1)]})
    return out


def frame_cooccurrence(frame_stack, min_px: int = 3) -> list:
    """For each frame transition, the regions that change TOGETHER in that frame.

    Returns [{frame, regions:[comp...], n_regions}] only for frames where >=2 distinct
    regions changed -- those are the candidate RELATED sets.  Same-frame change ⇒ likely
    related (a designer's temporal binding)."""
    n = 0 if frame_stack is None else len(frame_stack)
    out = []
    for i in range(1, n):
        m = _changed_mask(frame_stack[i - 1], frame_stack[i])
        if m is None:
            continue
        comps = _components(m, min_px)
        if len(comps) >= 2:
            out.append({"frame": i, "regions": comps, "n_regions": len(comps)})
    return out


def correlation_pairs(cooc, region_label) -> list:
    """Map each frame's co-occurring regions to labels (region_label(comp)->str|None) and
    emit deduped related PAIRS: [{frame, a, b}].  Pairs with an unlabelled region are kept
    with the raw center so the caller can still bind by geometry.  Game-agnostic."""
    seen, pairs = set(), []
    for fr in cooc:
        labs = [(region_label(c) or f"@{c['center']}") for c in fr["regions"]]
        for ai in range(len(labs)):
            for bi in range(ai + 1, len(labs)):
                a, b = sorted((labs[ai], labs[bi]))
                if a == b:
                    continue
                key = (a, b)
                if key not in seen:
                    seen.add(key)
                    pairs.append({"frame": fr["frame"], "a": a, "b": b})
    return pairs
