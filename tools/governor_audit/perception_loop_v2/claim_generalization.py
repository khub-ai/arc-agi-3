"""claim_generalization.py -- find the CORRECT cross-instance claim by least-general generalization.

A cross-level / cross-game regularity (the WIN CONDITION is the motivating case) is NOT a fixed fact --
it is a CLAIM, held as the least-general generalization (LGG) of the instances that support it.  When a
new instance (a newly understood level) is NOT covered by the current claim, the claim is
RE-GENERALIZED so it covers ALL known instances -- as specific as possible while still covering
everything.  This avoids both a brittle per-instance patchwork (under-general -- "lc1 means X, lc3
means Y") and a vacuous "anything goes" claim (over-general).

Mechanism: anti-unification over feature dicts with an optional ABSTRACTION TAXONOMY.  Per feature, two
differing values generalize to their least common ancestor in the taxonomy (e.g. complementary,
matching -> compatible); with no common ancestor they widen to a bounded disjunction; an over-wide
disjunction collapses to ANY.  The substrate MEASURES coverage + computes the generalization; the actor
supplies the per-instance features + the taxonomy (the semantics).  Domain-agnostic.

Worked case (tn36 win condition):
  lc1, lc2 instances: {combine: yes, shape_relation: complementary, result: closed_box}
  lc3 instance:       {combine: yes, shape_relation: matching,      result: merged}
  -> induced claim:   {combine: yes, shape_relation: compatible,    result: unified_figure}
  which then PREDICTS lc4 (complementary arch caps the U -> closed box) without re-generalizing.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

ANY = "*"  # wildcard: this feature is unconstrained


class Taxonomy:
    """A child -> parent map over feature values; used to generalize to a least common ancestor."""

    def __init__(self, parents: Optional[Dict[str, str]] = None):
        self._parent: Dict[str, str] = dict(parents or {})

    def add(self, child: str, parent: str) -> None:
        self._parent[child] = parent

    def chain(self, v: str) -> List[str]:
        """v and its ancestors, nearest first (cycle-safe)."""
        out = [v]
        seen = {v}
        while v in self._parent:
            v = self._parent[v]
            if v in seen:
                break
            out.append(v)
            seen.add(v)
        return out

    def lca(self, a: str, b: str) -> Optional[str]:
        ca = self.chain(a)
        cb = set(self.chain(b))
        for x in ca:
            if x in cb:
                return x
        return None


def _as_set(v):
    return v if isinstance(v, frozenset) else frozenset([v])


def generalize_value(cur, new, tax: Optional[Taxonomy] = None, max_disjunction: int = 3):
    """Least-general value covering both `cur` (a constant / frozenset disjunction / ANY) and `new`."""
    if cur == ANY or new == ANY:
        return ANY
    cur_set = _as_set(cur)
    if new in cur_set:
        return cur  # already covered
    if tax is not None:
        # a single ancestor covering every current member AND new?
        acc = None
        members = list(cur_set) + [new]
        acc = members[0]
        for m in members[1:]:
            acc = tax.lca(acc, m)
            if acc is None:
                break
        if acc is not None:
            return acc
    widened = cur_set | {new}
    if len(widened) > max_disjunction:
        return ANY
    return frozenset(widened)


def covers_value(patval, val, tax: Optional[Taxonomy] = None) -> bool:
    if patval == ANY:
        return True
    pset = _as_set(patval)
    if val in pset:
        return True
    if tax is not None:
        for p in pset:
            if p in tax.chain(val):  # p is an ancestor of val
                return True
    return False


def covers(pattern: Dict, instance: Dict, tax: Optional[Taxonomy] = None) -> bool:
    """Does the pattern cover the instance?  Features absent from the pattern are ANY."""
    for k, v in instance.items():
        if k in pattern and not covers_value(pattern[k], v, tax):
            return False
    return True


def generalize(pattern: Dict, instance: Dict, tax: Optional[Taxonomy] = None,
               max_disjunction: int = 3) -> Dict:
    """LGG of a pattern and one instance.  A feature present in only one side is not a shared
    constraint -> ANY."""
    out: Dict = {}
    for k in set(pattern) | set(instance):
        if k in pattern and k in instance:
            out[k] = generalize_value(pattern[k], instance[k], tax, max_disjunction)
        else:
            out[k] = ANY
    return out


def induce(instances: List[Dict], tax: Optional[Taxonomy] = None, max_disjunction: int = 3) -> Dict:
    """Fold generalize over instances -> the least-general pattern covering all of them."""
    if not instances:
        return {}
    pattern = dict(instances[0])
    for inst in instances[1:]:
        pattern = generalize(pattern, inst, tax, max_disjunction)
    return pattern


def _fmt(v) -> str:
    if v == ANY:
        return "ANY"
    if isinstance(v, frozenset):
        return "{" + "|".join(sorted(v)) + "}"
    return str(v)


class GeneralizedClaim:
    """A cross-instance claim held as the LGG of its supporting instances; re-generalizes on a miss.

    FEATURE DISCOVERY + BACK-FILL: a feature can be RELEVANT yet invisible until a new instance varies
    it (e.g. colour-uniformity is latent across lc1-3 -- the pieces were already one colour -- and only
    becomes salient at lc4, which needs a recolour).  A feature present in only SOME instances is not a
    shared constraint, so plain LGG drops it to ANY (losing it).  With a ``measure`` callback the claim
    RE-EXAMINES the prior instances for the newly-seen feature and back-fills it, so a genuinely-constant
    feature is KEPT in the claim -- letting COS refine "one solid block" into "one solid block of the
    same colour" on its own.  ``pending_backfill`` lists discovered-but-unaligned features when there is
    no callback.
    """

    def __init__(self, name: str, tax: Optional[Taxonomy] = None, max_disjunction: int = 3,
                 measure=None):
        self.name = name
        self.tax = tax
        self.max_disjunction = max_disjunction
        self.measure = measure  # optional measure(instance_id, feature) -> value (for back-fill)
        self.pattern: Optional[Dict] = None
        self.instances: List[Tuple[str, Dict]] = []
        self.generalizations: List[Dict] = []  # log of each forced re-generalization

    def _recompute(self) -> None:
        self.pattern = (induce([f for _, f in self.instances], self.tax, self.max_disjunction)
                        if self.instances else None)

    def _auto_backfill(self, new_features) -> None:
        """For each feature the new instance has that a PRIOR instance lacks, re-measure it for that
        prior via the callback so a latent-but-constant feature isn't dropped as non-shared."""
        keys = set(new_features)
        for i, (iid, f) in enumerate(self.instances[:-1]):
            for k in keys:
                if k not in f:
                    try:
                        v = self.measure(iid, k)
                    except Exception:
                        v = None
                    if v is not None:
                        f[k] = v
            self.instances[i] = (iid, f)

    def add(self, instance_id: str, features: Dict) -> bool:
        """Register an instance.  Returns True if it was already COVERED (the claim held), False if it
        FORCED a generalization (the claim had to widen to cover it)."""
        covered = self.pattern is not None and covers(self.pattern, features, self.tax)
        before = dict(self.pattern) if self.pattern is not None else None
        self.instances.append((instance_id, dict(features)))
        if self.measure is not None:
            self._auto_backfill(set(features))
        self._recompute()
        if before is not None and not covered and self.pattern != before:
            self.generalizations.append({"on": instance_id, "from": before, "to": dict(self.pattern)})
        return covered

    def pending_backfill(self) -> List[str]:
        """Features present in SOME but not ALL instances -- discovered features awaiting back-fill (so
        the actor can re-measure them across prior instances and call ``backfill``)."""
        if not self.instances:
            return []
        allkeys: set = set().union(*[set(f) for _, f in self.instances])
        return sorted(k for k in allkeys
                      if 0 < sum(1 for _, f in self.instances if k in f) < len(self.instances))

    def backfill(self, feature: str, values: Dict[str, object]) -> None:
        """Supply a discovered feature's value for prior instances (``{instance_id: value}``) and
        re-induce, so the claim keeps it."""
        for i, (iid, f) in enumerate(self.instances):
            if iid in values and feature not in f:
                f[feature] = values[iid]
                self.instances[i] = (iid, f)
        self._recompute()

    def predicts(self, features: Dict) -> bool:
        """Would the current claim cover this (hypothetical) instance?"""
        return self.pattern is not None and covers(self.pattern, features, self.tax)

    def covers_all(self) -> bool:
        return all(covers(self.pattern, f, self.tax) for _, f in self.instances)

    def summary(self) -> str:
        if self.pattern is None:
            return f"[claim {self.name}] (no instances)"
        body = ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(self.pattern.items()))
        s = (f"[claim {self.name}] {body}  "
             f"(covers {len(self.instances)} instance(s); {len(self.generalizations)} generalization(s))")
        return s
