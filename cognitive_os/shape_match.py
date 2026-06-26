"""Layer 3 of the visual-event perception stack: shape matching.

Pure functions operating on 2-D ``Pattern`` tuples. No world-state,
no config, no hypotheses, no miners.

Principle P4 lives here: the single public primitive is
:func:`shape_compare`, which returns a structured
:class:`ShapeComparison` describing *every* way two patterns relate
(identical, silhouette-equal, rotation, reflection, integer scale,
colour-remap). Callers inspect the fields they care about. We do not
expose boolean helpers like ``shape_equal_under_rotation`` — if a
consumer wants "equal under rotation," it's a one-line check against
``cmp.rotation_deg``.

Transform vocabulary covered in v1:

* **Rotations:** 0°, 90°, 180°, 270° (CCW).
* **Reflections:** horizontal axis, vertical axis, both.
* **Integer scale:** factor ``(sr, sc)`` with ``sr, sc >= 1`` in each
  axis (separable). Each cell of the smaller pattern maps to an
  ``sr × sc`` block of the larger.
* **Colour remap:** bijective per-value mapping on non-background
  cells when both patterns share the same silhouette.

Out of scope for v1: non-integer scale, skew, affine, perspective.
See ``SPEC_visual_event_perception.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional, Sequence, Tuple


Pattern = Tuple[Tuple[Any, ...], ...]


@dataclass(frozen=True)
class ShapeComparison:
    """Structured result of comparing two patterns.

    Every field reports independently on one relationship between the
    two patterns. Multiple fields may be set simultaneously — e.g. two
    patterns that are both rotation-related and silhouette-equal.

    Fields
    ------
    identical
        ``True`` iff the two patterns have the same shape (rows,
        cols) and the same cell values at every position.
    silhouette
        ``True`` iff the two patterns, treated as occupied/unoccupied
        masks against the supplied ``background``, have the same
        occupancy pattern at corresponding positions. Always ``False``
        if ``background`` was not supplied.
    rotation_deg
        The angle (one of 0, 90, 180, 270) that rotates pattern ``a``
        into pattern ``b`` exactly (cell-for-cell), or ``None`` if no
        pure rotation relates them. ``0`` means identical without any
        rotation; ``rotation_deg=0`` coexists with ``identical=True``.
    reflection
        ``"horizontal"``, ``"vertical"``, ``"both"``, or ``None``.
        ``"horizontal"`` means ``a`` flipped about its horizontal axis
        (row 0 ↔ last row) equals ``b``. ``"both"`` means the
        combined horizontal+vertical flip equals ``b`` but neither
        single flip does on its own.
    scale
        ``(sr, sc)`` integer factor mapping the smaller pattern to the
        larger one: ``b.rows == a.rows * sr`` (or the inverse) and each
        source cell expands to an ``sr × sc`` block. ``(1, 1)``
        co-exists with ``identical=True``. ``None`` if no pure integer
        scale relates them.
    colour_remap
        If the two silhouettes match (cell-by-cell occupancy equal
        against the background) and a bijection over non-background
        values relates their palettes, the mapping ``a_value →
        b_value`` is returned. ``None`` if silhouettes don't match,
        palettes are incompatible, or background was not supplied.
    invariances_needed
        The set (from ``{"rotation", "reflection", "scale",
        "colour"}``) of transforms that had to be tolerated for any
        non-trivial relationship to hold. Empty iff
        ``identical=True``.
    """
    identical:          bool
    silhouette:         bool
    rotation_deg:       Optional[int]
    reflection:         Optional[str]
    scale:              Optional[Tuple[int, int]]
    colour_remap:       Optional[Dict[Any, Any]]
    invariances_needed: FrozenSet[str]


_SENTINEL = object()


def shape_compare(
    a: Sequence[Sequence[Any]],
    b: Sequence[Sequence[Any]],
    *,
    background: Any = _SENTINEL,
) -> ShapeComparison:
    """Return every way ``a`` and ``b`` are related.

    ``background`` is needed for silhouette and colour-remap
    computation; omit it and those fields remain ``None`` /
    ``False``. Rotation, reflection, scale, and identity do not need
    a background — they operate on raw cell values.
    """
    a = _freeze(a)
    b = _freeze(b)

    has_bg = background is not _SENTINEL

    identical = (a == b)

    rotation_deg = _find_rotation(a, b)
    reflection   = _find_reflection(a, b)
    scale_factor = _find_scale(a, b)

    silhouette   = False
    colour_remap: Optional[Dict[Any, Any]] = None
    if has_bg:
        silhouette = _silhouette_equal(a, b, background)
        if silhouette:
            colour_remap = _colour_remap(a, b, background)

    invariances = set()
    if rotation_deg is not None and rotation_deg != 0:
        invariances.add("rotation")
    if reflection is not None:
        invariances.add("reflection")
    if scale_factor is not None and scale_factor != (1, 1):
        invariances.add("scale")
    if colour_remap is not None and any(k != v for k, v in colour_remap.items()):
        invariances.add("colour")

    return ShapeComparison(
        identical          = identical,
        silhouette         = silhouette,
        rotation_deg       = rotation_deg,
        reflection         = reflection,
        scale              = scale_factor,
        colour_remap       = colour_remap,
        invariances_needed = frozenset(invariances),
    )


# ---------------------------------------------------------------------------
# Pattern transforms (pure)
# ---------------------------------------------------------------------------


def rotate_pattern(p: Sequence[Sequence[Any]], degrees: int) -> Pattern:
    """Rotate ``p`` counter-clockwise by ``degrees`` (0/90/180/270)."""
    p = _freeze(p)
    deg = degrees % 360
    if deg == 0:
        return p
    if deg == 90:
        if not p:
            return p
        cols = len(p[0])
        return tuple(tuple(row[cols - 1 - i] for row in p) for i in range(cols))
    if deg == 180:
        return tuple(tuple(reversed(row)) for row in reversed(p))
    if deg == 270:
        if not p:
            return p
        cols = len(p[0])
        return tuple(tuple(row[i] for row in reversed(p)) for i in range(cols))
    raise ValueError(f"rotate_pattern requires a multiple of 90; got {degrees}")


def reflect_pattern(p: Sequence[Sequence[Any]], axis: str) -> Pattern:
    """Reflect ``p``. ``axis`` is ``"horizontal"``, ``"vertical"``, or ``"both"``.

    ``"horizontal"`` flips row 0 with the last row (top-bottom).
    ``"vertical"`` flips column 0 with the last column (left-right).
    ``"both"`` composes the two (equivalent to a 180° rotation).
    """
    p = _freeze(p)
    if axis == "horizontal":
        return tuple(reversed(p))
    if axis == "vertical":
        return tuple(tuple(reversed(row)) for row in p)
    if axis == "both":
        return tuple(tuple(reversed(row)) for row in reversed(p))
    raise ValueError(f"reflect_pattern axis must be horizontal|vertical|both; got {axis!r}")


def scale_pattern(p: Sequence[Sequence[Any]], sr: int, sc: int) -> Pattern:
    """Nearest-neighbour integer scale: each cell → an ``sr×sc`` block."""
    if sr < 1 or sc < 1:
        raise ValueError(f"scale factors must be >= 1; got ({sr}, {sc})")
    p = _freeze(p)
    out: list = []
    for row in p:
        expanded = tuple(v for v in row for _ in range(sc))
        for _ in range(sr):
            out.append(expanded)
    return tuple(out)


def silhouette(p: Sequence[Sequence[Any]], background: Any) -> Pattern:
    """Project ``p`` into a 0/1 silhouette against ``background``."""
    p = _freeze(p)
    return tuple(tuple(0 if v == background else 1 for v in row) for row in p)


# ---------------------------------------------------------------------------
# Internals — relationship checks
# ---------------------------------------------------------------------------


def _freeze(p: Sequence[Sequence[Any]]) -> Pattern:
    if isinstance(p, tuple) and (not p or isinstance(p[0], tuple)):
        return p  # type: ignore[return-value]
    return tuple(tuple(row) for row in p)


def _shape(p: Pattern) -> Tuple[int, int]:
    rows = len(p)
    cols = len(p[0]) if rows else 0
    return rows, cols


def _find_rotation(a: Pattern, b: Pattern) -> Optional[int]:
    for deg in (0, 90, 180, 270):
        if rotate_pattern(a, deg) == b:
            return deg
    return None


def _find_reflection(a: Pattern, b: Pattern) -> Optional[str]:
    for axis in ("horizontal", "vertical", "both"):
        if reflect_pattern(a, axis) == b:
            return axis
    return None


def _find_scale(a: Pattern, b: Pattern) -> Optional[Tuple[int, int]]:
    ar, ac = _shape(a)
    br, bc = _shape(b)
    if ar == 0 or ac == 0 or br == 0 or bc == 0:
        return None

    if ar == br and ac == bc:
        return (1, 1) if a == b else None

    if br >= ar and bc >= ac and br % ar == 0 and bc % ac == 0:
        sr, sc = br // ar, bc // ac
        if scale_pattern(a, sr, sc) == b:
            return (sr, sc)
        return None

    if ar >= br and ac >= bc and ar % br == 0 and ac % bc == 0:
        sr, sc = ar // br, ac // bc
        if scale_pattern(b, sr, sc) == a:
            return (sr, sc)
        return None

    return None


def _silhouette_equal(a: Pattern, b: Pattern, background: Any) -> bool:
    if _shape(a) != _shape(b):
        return False
    return silhouette(a, background) == silhouette(b, background)


def _colour_remap(a: Pattern, b: Pattern, background: Any) -> Optional[Dict[Any, Any]]:
    """Return bijection on non-background values, or ``None``.

    Assumes ``_silhouette_equal(a, b, background)`` already holds.
    """
    forward: Dict[Any, Any] = {}
    backward: Dict[Any, Any] = {}
    for row_a, row_b in zip(a, b):
        for va, vb in zip(row_a, row_b):
            if va == background and vb == background:
                continue
            if va == background or vb == background:
                return None  # silhouette mismatch (shouldn't happen given precondition)
            if va in forward:
                if forward[va] != vb:
                    return None
            else:
                forward[va] = vb
            if vb in backward:
                if backward[vb] != va:
                    return None
            else:
                backward[vb] = va
    return forward
