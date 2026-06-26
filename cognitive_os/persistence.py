"""Cross-episode knowledge persistence — save/load committed claims.

Rationale
---------
A cognitive OS whose hypotheses are discarded at episode end has no
way to accumulate knowledge.  The same action-probing work, the same
region-motion extraction, the same controlled-actor deduction would
repeat from scratch on every attempt.  This module closes that loop:
at the end of every episode the committed, **transferable** claims
are written to disk, and at the start of the next episode they are
loaded back into the hypothesis store with their earned credence.

What "transferable" means
-------------------------
A claim is transferable iff its canonical key would remain meaningful
in a future episode started from a different position with a
different flood-fill entity labelling.  The types currently
whitelisted meet that bar by construction:

* :class:`ControlledActorClaim(colour, background)` — keyed by a
  pure colour signature.  Same sprite on same background in any
  future run produces the same key.
* :class:`CausalClaim(ActionJustTaken(action_id), RegionMotion(...))`
  — action id is the adapter's stable string, and
  :class:`RegionMotion` has sign-only direction components.  Together
  they encode "action X moves sprite with colour Y on background Z
  in direction (dr_sign, dc_sign)", with no absolute coordinates.
* :class:`BitmapRoleClaim(bitmap_id, role)` — keyed by a canonical
  sprite-bitmap fingerprint.  Bitmap_ids are stable hashes computed
  from sprite pixels; the same sprite produces the same key across
  episodes / levels of one game.  Added as part of the
  context_memory migration (see docs/SPEC_context_memory_component.md)
  to bridge perception-substrate role assignments into the engine's
  hypothesis store.
* :class:`RegionPaletteClaim(palettes, role)` — keyed by a palette
  signature (sorted-deduped tuple of palette integers).  Palette
  indices are stable within an env across levels.  Same migration.

Explicitly **not** persisted:

* :class:`PropertyClaim` — keyed by ephemeral entity ids that don't
  survive a new flood-fill pass.
* :class:`CausalClaim` with :class:`FrameChangedPattern` effect — the
  bounding box is absolute and episode-local.
* :class:`TransitionClaim` between :class:`AtPosition` conditions —
  positions are absolute.

As new transferable claim forms are invented (e.g. structural
mappings, learned options), this module's whitelist extends; the
architecture itself does not.

File layout
-----------
A single JSONL file per ``knowledge_dir``::

    knowledge_dir/claims.jsonl

One line per persisted claim, each a self-contained JSON object with
a ``claim_type`` discriminator used by :func:`load_committed_knowledge`
to dispatch to the correct reconstructor.  JSONL is the right format
for append-friendly, diff-friendly, git-friendly storage.

Domain agnosticism
------------------
The module is engine-side and domain-agnostic: nothing in
``cognitive_os`` depends on any adapter.  Callers (ARC harnesses,
robotics harnesses) provide the ``knowledge_dir`` path — typically
one directory per environment id, so that ``ls20`` and ``ls21``
accumulate separately — and the engine does the rest.

Capability audit
----------------
* **Problem-solving** — PRIMARY.  Without persistence, a one-shot
  agent rediscovers its own action semantics every run, burning
  steps that should have gone to planning.
* **Debugging**        — secondary.  A persisted claim history is
  the cleanest possible per-environment audit trail; grep-friendly
  by design.
* **Tool creation**    — secondary.  Once :class:`Option` synthesis
  lands (Phase 7), Options persist through this same substrate —
  the only change is a new case in the whitelist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .claims import (
    BitmapRoleClaim,
    CausalClaim,
    ControlledActorClaim,
    RegionPaletteClaim,
)
from .conditions import ActionJustTaken, RegionMotion
from . import hypothesis_store as _store
from .types import Scope, ScopeKind, WorldState


_KNOWLEDGE_FILENAME = "claims.jsonl"


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_committed_knowledge(ws:             WorldState,
                             knowledge_dir:  Union[str, Path]) -> List[Dict[str, Any]]:
    """Serialise every committed transferable claim in ``ws`` to
    ``{knowledge_dir}/claims.jsonl``.

    Overwrites the file.  The caller is responsible for choosing a
    ``knowledge_dir`` scoped appropriately to the environment — e.g.
    ``.knowledge/ls20/``.  Mixing environments in one directory would
    re-propose ls20's directional claims into an ls21 episode, which
    is exactly wrong even if the claim shapes are compatible.

    Returns the list of record dicts actually written, so callers can
    log / test / audit without re-reading the file.

    Only claims whose ``full_key`` matches one of the transferable
    shapes are written.  The point is *not* to round-trip everything
    in the store — it's to transfer only the claims whose canonical
    keys are stable across episodes.
    """
    path = Path(knowledge_dir)
    path.mkdir(parents=True, exist_ok=True)
    out_path = path / _KNOWLEDGE_FILENAME

    records: List[Dict[str, Any]] = []
    for h in _store.committed(ws):
        rec = _encode_claim(h)
        if rec is None:
            continue  # not on the whitelist
        records.append(rec)

    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True))
            fh.write("\n")

    return records


def _encode_claim(h) -> Optional[Dict[str, Any]]:
    """Return the JSON-serialisable record for a committed hypothesis,
    or ``None`` if the claim is not on the transferable whitelist.

    Every record carries:

    * ``claim_type`` — discriminator for the loader.
    * ``source``, ``scope`` — provenance, useful for audit and for
      load-time filtering (e.g. only load ``GAME``-scoped claims).
    * ``credence``, ``evidence_weight`` — the earned confidence.  The
      loader re-proposes the claim with ``credence`` as its initial
      point, preserving the conviction the episode accumulated.

    The per-type fields are flat and primitive so the JSONL is
    human-readable.
    """
    claim    = h.claim
    common   = {
        "source":           h.source,
        "scope":            h.scope.kind.value if h.scope is not None else None,
        "credence":         float(h.credence.point),
        "evidence_weight":  float(h.credence.evidence_weight),
        "supports":         len(h.supporting_steps),
    }

    if isinstance(claim, ControlledActorClaim):
        return {
            "claim_type": "ControlledActorClaim",
            "colour":     claim.colour,
            "background": claim.background,
            **common,
        }

    if (isinstance(claim, CausalClaim)
            and isinstance(claim.trigger, ActionJustTaken)
            and isinstance(claim.effect,  RegionMotion)):
        return {
            "claim_type": "CausalClaim:ActionJustTaken:RegionMotion",
            "action_id":  claim.trigger.action_id,
            "colour":     claim.effect.colour,
            "background": claim.effect.background,
            "dr_sign":    int(claim.effect.dr_sign),
            "dc_sign":    int(claim.effect.dc_sign),
            **common,
        }

    if isinstance(claim, BitmapRoleClaim):
        rec: Dict[str, Any] = {
            "claim_type": "BitmapRoleClaim",
            "bitmap_id":  str(claim.bitmap_id),
            "role":       str(claim.role),
            **common,
        }
        # Optional matcher metadata.  Omitted from the record when
        # None so legacy JSONL files stay diff-clean.
        if claim.shape_id is not None:
            rec["shape_id"] = str(claim.shape_id)
        if claim.topo_id is not None:
            rec["topo_id"] = str(claim.topo_id)
        if claim.size_px is not None:
            rec["size_px"] = int(claim.size_px)
        if claim.spatial_zone is not None:
            rec["spatial_zone"] = str(claim.spatial_zone)
        return rec

    if isinstance(claim, RegionPaletteClaim):
        rec = {
            "claim_type": "RegionPaletteClaim",
            "palettes":   list(claim.palettes),   # JSON-friendly
            "role":       str(claim.role),
            **common,
        }
        if claim.row_range is not None:
            rec["row_range"] = [int(claim.row_range[0]),
                                int(claim.row_range[1])]
        if claim.spatial_zone is not None:
            rec["spatial_zone"] = str(claim.spatial_zone)
        return rec

    return None


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_committed_knowledge(ws:            WorldState,
                             knowledge_dir: Union[str, Path],
                             *,
                             step: int = 0) -> List[str]:
    """Re-propose every persisted claim into ``ws``'s hypothesis store.

    Missing file is not an error — a fresh environment has nothing to
    load; the runner simply proceeds and the next save seeds the
    file.  Malformed lines are skipped with no exception so that a
    single corrupt record can't poison an otherwise valid knowledge
    base.

    Each loaded claim is proposed at its persisted ``credence``
    (overriding source priors) and at ``step`` (default ``0``, i.e.
    treat the claim as having been in the store since before the
    episode began).  This keeps committed claims *already committed*
    the moment load finishes — no re-probing required.

    Returns the list of hypothesis ids created.
    """
    path = Path(knowledge_dir) / _KNOWLEDGE_FILENAME
    if not path.exists():
        return []

    ids: List[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            h_id = _reconstruct_and_propose(ws, rec, step=step)
            if h_id is not None:
                ids.append(h_id)
    return ids


def _reconstruct_and_propose(ws:   WorldState,
                             rec:  Dict[str, Any],
                             *,
                             step: int) -> Optional[str]:
    """Build a Claim from ``rec`` and propose it into ``ws``.

    Silent-skip on anything unrecognised: the loader must stay robust
    across schema evolutions.  When a new claim type lands, this
    function grows a case; old records for a retired type become
    no-ops on load.
    """
    claim_type = rec.get("claim_type")
    source     = rec.get("source")     or "engine:persisted"
    credence   = rec.get("credence")
    scope_kind = rec.get("scope")
    scope      = _scope_from_kind_name(scope_kind)

    if claim_type == "ControlledActorClaim":
        claim = ControlledActorClaim(
            colour     = rec.get("colour"),
            background = rec.get("background"),
        )
    elif claim_type == "CausalClaim:ActionJustTaken:RegionMotion":
        claim = CausalClaim(
            trigger = ActionJustTaken(action_id=str(rec.get("action_id"))),
            effect  = RegionMotion(
                colour     = rec.get("colour"),
                background = rec.get("background"),
                dr_sign    = int(rec.get("dr_sign", 0)),
                dc_sign    = int(rec.get("dc_sign", 0)),
            ),
        )
    elif claim_type == "BitmapRoleClaim":
        size_raw = rec.get("size_px")
        claim = BitmapRoleClaim(
            bitmap_id    = str(rec.get("bitmap_id", "")),
            role         = str(rec.get("role", "")),
            shape_id     = (str(rec["shape_id"])
                            if rec.get("shape_id") is not None else None),
            topo_id      = (str(rec["topo_id"])
                            if rec.get("topo_id")  is not None else None),
            size_px      = (int(size_raw) if size_raw is not None else None),
            spatial_zone = (str(rec["spatial_zone"])
                            if rec.get("spatial_zone") is not None else None),
        )
    elif claim_type == "RegionPaletteClaim":
        # ``make`` normalises arbitrary input to the canonical
        # sorted-deduped tuple form so a hand-edited or older record
        # with an unsorted palette list still round-trips correctly.
        rr_raw = rec.get("row_range")
        rr: Optional[Tuple[int, int]] = None
        if isinstance(rr_raw, (list, tuple)) and len(rr_raw) == 2:
            rr = (int(rr_raw[0]), int(rr_raw[1]))
        claim = RegionPaletteClaim.make(
            palettes     = rec.get("palettes") or (),
            role         = str(rec.get("role", "")),
            row_range    = rr,
            spatial_zone = (str(rec["spatial_zone"])
                            if rec.get("spatial_zone") is not None else None),
        )
    else:
        return None

    return _store.propose(
        ws,
        claim            = claim,
        source           = source,
        scope            = scope,
        step             = step,
        initial_credence = (float(credence) if credence is not None else None),
        rationale        = "persisted",
    )


def _scope_from_kind_name(name: Optional[str]) -> Scope:
    """Parse a stored scope kind back into a :class:`Scope`.

    Only the kind is persisted — the narrower filters
    (``position_region``, ``entity_filter``, ``time_range``) are
    either absolute (hence non-transferable) or evaluated at reload
    time; either way they are not restored here.  Callers whose
    knowledge needs richer scope preservation can extend the record
    schema.
    """
    if name is None:
        return Scope(kind=ScopeKind.GAME)
    try:
        return Scope(kind=ScopeKind(name))
    except ValueError:
        return Scope(kind=ScopeKind.GAME)
