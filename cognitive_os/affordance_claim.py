"""AffordanceClaim — a typed hypothesis that a visual class affords a
typed effect under a typed trigger geometry.

The claim is the cross-game-transferable unit: it carries no level or
absolute-cell tokens, only the visual class fingerprint plus an AST
of primitives from :mod:`cognitive_os.affordance_dsl`.  Per-game,
per-instance, per-frame application is the resolver's job (see
:mod:`cognitive_os.affordance_resolver`).

The claim sits alongside :class:`cognitive_os.claims.CausalClaim` and
participates in the same hypothesis-store accumulation / credence
machinery.

See :doc:`docs/SPEC_entity_affordances.md` for the full design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .affordance_dsl import Node


@dataclass(frozen=True, eq=False)
class AffordanceClaim:
    """A visual class affords a typed effect under a typed trigger.

    ``entity_class`` is a string fingerprint at one of the three tiers
    used elsewhere in the system (``bitmap_id`` / ``shape_id`` /
    ``topo_id``).  Two instances whose fingerprint matches at the same
    tier as the claim's are considered the same class for resolution.

    ``trigger`` and ``effect`` are AST nodes from
    :mod:`cognitive_os.affordance_dsl`.  Together they describe the
    affordance as a *composition* over the closed primitive
    vocabularies, not an enumeration label.

    ``labels`` is an open dict of advisory metadata — telemetry,
    prior-import provenance, human readability.  No runtime decision
    branches on label content.
    """
    entity_class: str
    trigger:      Node
    effect:       Node
    labels:       Mapping[str, str] = field(default_factory=dict)

    def canonical_key(self) -> tuple:
        """Hashable structural identity of the claim.  Two claims with
        the same key are duplicates.  Labels are advisory and do NOT
        participate in identity."""
        return (
            "AffordanceClaim",
            str(self.entity_class),
            self.trigger.canonical_key(),
            self.effect.canonical_key(),
        )

    def __hash__(self) -> int:
        return hash(self.canonical_key())

    def __eq__(self, other) -> bool:
        if not isinstance(other, AffordanceClaim):
            return False
        return self.canonical_key() == other.canonical_key()
