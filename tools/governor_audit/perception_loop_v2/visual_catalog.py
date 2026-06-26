"""visual_catalog.py -- COS's cross-game VISUAL MEMORY of salient entities.

A good visual memory: when COS sees a salient entity (a legend icon, a glyph, a distinctive sprite) it
should recognise whether it has seen something LIKE it before -- in this game OR ANOTHER -- and recall
what that thing MEANT, as a PRIOR.  So a new game's icon that resembles a known one inherits a good
guess about its function before any probing (e.g. a yellow-diamond/4-cross icon recurs across games as a
ROTATE control).

Each catalog entry pairs a VISUAL SIGNATURE (the rotation/scale-invariant shape signature from
shape_identity, plus an identifying colour and a small crop for display) with the MEANING learned for it
(what it is / what it does), the games + levels it was seen in, a credence, and provenance.  Matching
uses shape_identity.similarity (pose-invariant; no tuned threshold) with a soft colour-agreement factor,
so a re-coloured or rotated re-skin of a known icon still matches.  Persisted in the unified KB, so the
memory is genuinely CROSS-GAME and portable.

The catalog MEASURES resemblance and surfaces candidates with their scores; the acting VLM decides
whether to trust the prior -- shape similarity is shown, never silently thresholded.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from shape_identity import similarity


def _default_path():
    try:
        from kb_paths import kb_path
        return kb_path("visual_catalog.json")
    except Exception:
        return None


@dataclass
class CatalogEntry:
    name: str                     # a short label for the entity (e.g. "diamond_rotate")
    sig: str                      # rotation/scale-invariant shape signature (shape_identity)
    color: str = ""               # identifying colour (hex), if known
    size: int = 0                 # foreground pixel count, rough scale
    crop_b64: str = ""            # small PNG crop for human/VLM visual compare
    meaning: str = ""             # what it IS (role / description)
    function: str = ""            # what it DOES (the learned effect), if known
    games: List[str] = field(default_factory=list)
    levels: List[str] = field(default_factory=list)
    credence: float = 0.5
    provenance: str = "observed"


class VisualCatalog:
    def __init__(self, path=None):
        self.path = Path(path) if path else _default_path()
        self.entries: List[CatalogEntry] = []
        if self.path and self.path.exists():
            try:
                self.entries = [CatalogEntry(**e) for e in json.loads(self.path.read_text(encoding="utf-8"))]
            except Exception:
                self.entries = []

    # ---- registration -----------------------------------------------------
    def record(self, name, sig, color="", size=0, crop_b64="", meaning="", function="",
               game="", level="", credence=0.6, provenance="learned") -> Optional[CatalogEntry]:
        """Catalog a salient entity (or update it).  Dedup by (name, game): re-seeing the SAME entity
        in the SAME game updates that record; the SAME-looking entity in a DIFFERENT game is a separate
        entry, so cross-game matches aggregate.  No-op if the signature is empty."""
        if not sig:
            return None
        for e in self.entries:
            if e.name == name and game in e.games:
                if meaning:
                    e.meaning = meaning
                if function:
                    e.function = function
                if color and not e.color:
                    e.color = color
                if crop_b64 and not e.crop_b64:
                    e.crop_b64 = crop_b64
                e.credence = max(e.credence, credence)
                if level and level not in e.levels:
                    e.levels.append(level)
                self._save()
                return e
        rec = CatalogEntry(name=name, sig=sig, color=color, size=int(size), crop_b64=crop_b64,
                           meaning=meaning, function=function, games=[game] if game else [],
                           levels=[level] if level else [], credence=credence, provenance=provenance)
        self.entries.append(rec)
        self._save()
        return rec

    def _save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([asdict(e) for e in self.entries], indent=2),
                                 encoding="utf-8")
        except Exception:
            pass

    # ---- recall -----------------------------------------------------------
    def match(self, sig, color: Optional[str] = None, exclude_game: Optional[str] = None,
              k: int = 3, admit=None) -> List[Tuple[CatalogEntry, float]]:
        """Catalog entries whose SHAPE resembles ``sig``, ranked by similarity (with a soft colour-
        agreement factor).  Returns (entry, score) pairs with score > 0, best first.  ``exclude_game``
        drops same-game entries so cross-game priors stand out.  No absolute cutoff -- the caller sees
        the scores and decides.

        ``admit`` is an optional cross-game recall gate: a callable
        ``admit(entry.games) -> bool``.  When given, an entry is dropped unless it passes - the caller
        uses cross_game_knowledge.admits_xgame so a prior from ANOTHER game surfaces only when that game
        is a structural VARIANT of the current one (else an unseen game would inherit a public game's
        sprite role).  None = no gating (registration / tests)."""
        if not sig:
            return []
        scored = []
        for e in self.entries:
            if exclude_game and e.games == [exclude_game]:
                continue
            if admit is not None and not admit(e.games):
                continue
            s = similarity(sig, e.sig)
            if s <= 0:
                continue
            if color and e.color:
                s *= 1.0 if color.lower() == e.color.lower() else 0.85   # shape dominates; colour nudges
            scored.append((e, round(s * e.credence, 4)))
        scored.sort(key=lambda t: -t[1])
        return scored[:k]

    def recall_prior(self, sig, color: Optional[str] = None, game: Optional[str] = None,
                     k: int = 2, admit=None) -> Optional[str]:
        """A PRIOR directive for a freshly-seen salient entity: name the best-matching catalog entries
        (cross-game first), each with its shape-similarity score and what it did before, so the actor
        can adopt a good guess about this entity BEFORE probing.  ``None`` if nothing resembles it.
        ``admit`` is the cross-game recall gate (see match)."""
        hits = self.match(sig, color=color, k=k, admit=admit)
        if not hits:
            return None
        lines = ["[VISUAL-MEMORY] this entity RESEMBLES one(s) seen before -- a PRIOR (verify, don't "
                 "assume): use the recalled function as a starting hypothesis:"]
        for e, score in hits:
            where = ("/".join(e.games) or "?")
            cross = " (ANOTHER game)" if game and e.games and game not in e.games else ""
            does = f" -> does: {e.function}" if e.function else (f" -> is: {e.meaning}" if e.meaning else "")
            lines.append(f"  - resembles '{e.name}' [shape-sim {score:.2f}; seen in {where}{cross}]{does}")
        return "\n".join(lines)
