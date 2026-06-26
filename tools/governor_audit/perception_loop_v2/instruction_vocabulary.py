"""instruction_vocabulary.py -- remember which CODE produces which EFFECT, learned from
a demonstration, so COS need not re-invoke the demonstration every time.

When follow-the-instructions decodes an option once (activate it, watch the demonstrated
effect, read the reference code), it LEARNs effect->code here.  Programming a goal then
just LOOKs UP each step's effect -- no repeated animations.  Game-agnostic and persistable
(any 'option -> encoding' vocabulary, e.g. direction->switch-pattern, key->action).
"""
from __future__ import annotations

from typing import Optional


def _norm_code(code):
    """Canonicalise a code to a sorted tuple so set/list orderings compare equal."""
    if isinstance(code, (set, frozenset, list, tuple)):
        return tuple(sorted(map(str, code)))
    return code


class InstructionVocabulary:
    def __init__(self, game: str = "") -> None:
        self.game = game
        self._b: dict = {}                                # effect(str) -> {code, provenance}

    def learn(self, effect, code, provenance: str = "demonstrated") -> None:
        self._b[str(effect)] = {"code": _norm_code(code), "provenance": provenance}

    def lookup(self, effect):
        e = self._b.get(str(effect))
        return e["code"] if e else None

    def knows(self, effect) -> bool:
        return str(effect) in self._b

    def known_effects(self) -> list:
        return sorted(self._b)

    def needed(self, effects) -> list:
        """Of the effects a goal needs, which are NOT yet learned (still need a demo)."""
        return [e for e in dict.fromkeys(map(str, effects)) if e not in self._b]

    # ---- persistence (per-game, reusable across runs) -----------------------
    def to_records(self) -> dict:
        return {k: {"code": list(v["code"]) if isinstance(v["code"], tuple) else v["code"],
                    "provenance": v["provenance"]} for k, v in self._b.items()}

    def merge_records(self, recs: Optional[dict]) -> None:
        for k, v in (recs or {}).items():
            self._b[str(k)] = {"code": _norm_code(v.get("code")),
                               "provenance": v.get("provenance", "loaded")}
