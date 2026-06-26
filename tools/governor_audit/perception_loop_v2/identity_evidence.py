"""IDENTITY evidence -- read a group's STATIC content before probing it.

The mistake this guards against: answering an IDENTITY question ("do these
similar elements differ / what do they ENCODE?") with FUNCTION evidence ("does
acting on one change anything?"), and treating a NON-discriminating result (no
visible change) as a NEGATIVE conclusion ("they're equivalent / not a key").

The identity/encoding of perceptual elements is a STATIC fact to be READ.  So,
for any group of similar/repeated elements (a candidate key / legend / control
set / palette), decode each member's static signature and report whether they
are mutually DISTINCT.  Distinct static content => the group encodes distinct
values (an index/key) -- regardless of any interaction-response.  Game-agnostic,
pure measurement; the consumer (the VLM) decides what the distinctions MEAN.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np


def static_signature(frame_rgb, bbox) -> Optional[dict]:
    """A member's COMPLETE static signature across every channel that can carry
    state -- so a partial read (one channel) cannot decide identity.  Channels:
      * shape    -- a coarse NxN spatial-colour grid (WHERE the colours sit:
                    orientation / arrangement / which corner a mark is in).
      * color    -- the SET of distinct colours present (catches a gray-LEVEL or
                    on/off state that a single 'dominant colour' washes out).
      * size     -- [h, w].
    N is adapted to the box size (bounded), a resolution choice -- not a tuned
    threshold.  None on bad input."""
    try:
        r0, c0, r1, c1 = [int(v) for v in bbox]
        reg = np.asarray(frame_rgb)[r0:r1 + 1, c0:c1 + 1, :3]
        h, w = reg.shape[:2]
        if h < 1 or w < 1:
            return None
        n = max(2, min(6, h, w))
        rr = np.linspace(0, h, n + 1)
        cc = np.linspace(0, w, n + 1)
        # shape = a COLOUR-INDEPENDENT mark-map: per sub-cell, 1 if it is not the
        # element's dominant (field) colour (i.e. it holds a mark), else 0.  So the
        # shape channel captures WHERE marks sit, orthogonal to WHICH colour.
        field = Counter(map(tuple, reg.reshape(-1, 3).tolist())).most_common(1)[0][0]
        shape = []
        for i in range(n):
            for j in range(n):
                sub = reg[int(rr[i]):max(int(rr[i]) + 1, int(rr[i + 1])),
                          int(cc[j]):max(int(cc[j]) + 1, int(cc[j + 1]))]
                dom = Counter(map(tuple, sub.reshape(-1, 3).tolist())).most_common(1)[0][0]
                shape.append(0 if dom == field else 1)
        color_set = tuple(sorted(set(map(tuple, reg.reshape(-1, 3).tolist()))))
        return {"grid_n": n, "cells": tuple(shape), "color_set": color_set, "size": [h, w]}
    except Exception:
        return None


_CHANNELS = ("shape", "color", "size")


def _channels(sig) -> Optional[dict]:
    if not sig or not sig.get("cells"):
        return None
    return {"shape": tuple(sig["cells"]), "color": sig.get("color_set"),
            "size": tuple(sig.get("size") or ())}


def group_distinctness(frame_rgb, members) -> dict:
    """``members``: list of {"name", "bbox"}.  Decode each member's COMPLETE static
    signature and report which CHANNELS distinguish them -- the IDENTITY evidence
    to read BEFORE any interaction-probe.

    Returns {"n_members", "n_distinct", "all_distinct", "differs_in", "members",
    "note"}.  ``differs_in`` names the channel(s) (shape | color | size) in which
    the members vary -- so a partial read (e.g. shape only, ignoring colour) is
    impossible: the substrate tells you which channel carries the difference.
    ``all_distinct`` True => the group encodes distinct values (a key/index); a
    click that produces no change can never make distinct members equivalent."""
    sigs = []
    for m in (members or []):
        sigs.append({"name": m.get("name"),
                     "signature": static_signature(frame_rgb, m.get("bbox"))})
    chans = [_channels(s["signature"]) for s in sigs]
    present = [c for c in chans if c is not None]
    differs_in = [ch for ch in _CHANNELS if len({c[ch] for c in present}) > 1]
    combined = [tuple(c[ch] for ch in _CHANNELS) for c in present]
    uniq = set(combined)
    note = ("Per-member signature across ALL channels (shape=where colours sit, "
            "color=the set of colour/gray levels present, size). ")
    if differs_in:
        note += (f"Members DIFFER in: {differs_in} -- read THOSE channel(s), not "
                 f"just the most salient one. Distinct signatures => the group "
                 f"ENCODES distinct values (a key/index); a null interaction-"
                 f"response can never make them equivalent.")
    else:
        note += ("Members are identical across all channels -- a uniform/repeated "
                 "set, not an index. (If you expected a distinction, it is not "
                 "static -- look for an interaction or a missing channel.)")
    return {"n_members": len(sigs), "n_distinct": len(uniq),
            "all_distinct": len(uniq) == len(present) and len(present) > 1,
            "differs_in": differs_in, "members": sigs, "note": note}
