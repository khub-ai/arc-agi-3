"""Bridge between the perception Scene and the cognitive_os
hypothesis_store.

Lets the VLM-driven perception loop commit credence-bearing claims
about tracks to the existing hypothesis_store, and stores the
returned hypothesis_ids back on the EntityTrack via annotate_entity.

Scope boundary
==============

The bridge maps a small set of VLM-proposable claim types to the
existing Claim subclasses in `cognitive_os.claims`.  The mapping
itself is straightforward; the architectural job is naming
convention and lifecycle:

  * Tracks have integer ids (track_id) in the registry; Claims
    reference entities by string id.  The bridge canonicalises
    the string id as "track:{track_id}" so downstream Claim
    consumers don't have to know about the perception layer.
  * The bridge does not invent its own credence machinery — it
    delegates to hypothesis_store.propose() which handles
    deduplication, competitor linking, and the full lifecycle.

What's supported now
====================

  propose_track_property(ws, scene, track_id, property, value)
    -> PropertyClaim about a track ("role" = "avatar", etc.)
    Returns the hypothesis_id; appends it to the track's
    hypothesis_ids list.

Future surface (stubs)
======================

  propose_track_causal(ws, scene, trigger_track, effect_track, ...)
    -> CausalClaim.  Requires Condition mapping from track-level
    facts ("agent enters track's cell") to the Condition vocabulary
    in cognitive_os.conditions.  Stub raises NotImplementedError;
    the work is design, not boilerplate.

  propose_track_transition, propose_track_relational
    -> Similar — deferred until the Condition mapping lands.

These stubs are present so the calling code (vlm_perception.py)
can prepare to call them; their implementation is a follow-up.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional


# We import from cognitive_os via sys.path -- the perception_loop_v2
# package is under tools/governor_audit, sibling to cognitive_os.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .temporal_registry import Scene  # noqa: E402
from . import vlm_tools as T  # noqa: E402


def _track_entity_id(track_id: int) -> str:
    """Canonical string id for a track in the hypothesis_store.
    Uses a 'track:' prefix so the namespace can coexist with the
    existing entity-id convention (numeric or alphanumeric)."""
    return f"track:{track_id}"


def propose_track_property(
    world_state: Any,
    scene: Scene,
    track_id: int,
    *,
    property: str,
    value: Any,
    source: str = "vlm:perception",
    step: int = 0,
    scope_kind: str = "episode",
    rationale: Optional[str] = None,
) -> Optional[str]:
    """Propose a PropertyClaim about a track, store the resulting
    hypothesis_id on the track's hypothesis_ids list, and return the id.

    Returns None if the WorldState / hypothesis_store machinery isn't
    available (import failure) — in that case the bridge degrades
    gracefully to a no-op so a caller without the full cognitive_os
    available can still run.

    Example:
        propose_track_property(
            ws, scene, track_id=1,
            property="role", value="avatar",
            step=turn, rationale="composite blue/yellow avatar sprite",
        )
    """
    try:
        from cognitive_os.claims import PropertyClaim
        from cognitive_os.hypothesis_store import propose
        from cognitive_os.types import Scope, ScopeKind
    except ImportError:
        # cognitive_os not importable in this context — degrade.
        return None
    claim = PropertyClaim(
        entity_id=_track_entity_id(track_id),
        property=property,
        value=value,
    )
    kind = ScopeKind[scope_kind.upper()] if hasattr(ScopeKind, scope_kind.upper()) else ScopeKind.EPISODE
    scope = Scope(kind=kind)
    hyp_id = propose(
        world_state, claim, source, scope, step,
        rationale=rationale,
    )
    if hyp_id is None:
        return None
    T.annotate_entity(scene, track_id, add_hypothesis_id=hyp_id)
    return hyp_id


def propose_track_causal(
    world_state: Any,
    scene: Scene,
    *,
    trigger_description: str,
    effect_track_id: int,
    effect_description: str,
) -> Optional[str]:
    """STUB.  Propose a CausalClaim involving a track.

    Not yet wired -- requires translating the VLM's free-text
    trigger/effect descriptions into the Condition vocabulary in
    cognitive_os.conditions (AtPosition, InsideBBox, EntityInState,
    ResourceAbove/Below, etc.).  That mapping is design work, not
    boilerplate; left as a follow-up.

    The signature is fixed so callers (vlm_perception.py) can stage
    causal proposals as comments without breaking; the call will
    log a deferred-implementation note and return None.
    """
    # Intentionally a graceful no-op for now.
    return None


def propose_track_transition(
    world_state: Any,
    scene: Scene,
    *,
    action: str,
    pre_track_id: int,
    pre_description: str,
    post_track_id: Optional[int],
    post_description: str,
) -> Optional[str]:
    """STUB. TransitionClaim: action with pre-state on a track leads
    to post-state.  Deferred for the same reason as causal claims --
    needs Condition mapping.
    """
    return None
