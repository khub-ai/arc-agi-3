"""Apply committed rules at perception time.

Given a FrameObservation and a RuleStore, produce hints that the
classifier consults BEFORE its per-frame derivation:

  - which quantised RGB key is the primary background (role W)
  - which is the secondary background (role B)
  - where the HUD strip lives (y-range)
  - the composite-sprite signature for hazards

The hints don't replace the per-frame algorithm — they bias it.  If
the rule says "primary background is colour X" and X exists in the
frame, the classifier uses X as the primary instead of just picking
the largest component (which might be different in some frames).
"""

from __future__ import annotations

from dataclasses import dataclass

from .rules import RuleStore, Rule


@dataclass
class RuleHints:
    """Hints derived from active rules, passed to the classifier."""
    primary_rgb_key: tuple[int, int, int] | None = None
    secondary_rgb_key: tuple[int, int, int] | None = None
    hud_y_range_logical: tuple[int, int] | None = None
    composite_color_set: list[tuple[int, int, int]] | None = None


def derive_hints(
    store: RuleStore,
    *,
    allow_sandbox: bool = False,
) -> RuleHints:
    """Return RuleHints from currently-active rules in the store."""
    hints = RuleHints()
    for rule in store.active(allow_sandbox=allow_sandbox):
        body = rule.body
        if rule.type == "color_binding":
            role = body.get("role")
            rgb = body.get("rgb_key")
            if rgb is None or not isinstance(rgb, (list, tuple)):
                continue
            rgb_t = tuple(int(v) for v in rgb)
            if role == "W" and hints.primary_rgb_key is None:
                hints.primary_rgb_key = rgb_t
            elif role == "B" and hints.secondary_rgb_key is None:
                hints.secondary_rgb_key = rgb_t
        elif rule.type == "hud_strip":
            if hints.hud_y_range_logical is None:
                yr = body.get("y_range_logical")
                if yr is not None and len(yr) == 2:
                    hints.hud_y_range_logical = (int(yr[0]), int(yr[1]))
        elif rule.type == "composite_sprite":
            if hints.composite_color_set is None:
                cs = body.get("color_set")
                if cs is not None:
                    hints.composite_color_set = [
                        tuple(int(v) for v in c) for c in cs
                    ]
    return hints
