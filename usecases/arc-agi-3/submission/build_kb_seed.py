"""Rebuild the competition KB seed (general / Tier-0 knowledge only) + a browsable
manifest of everything the agent knows at competition start.

WHY: in competition the KB is a fresh empty scratch dir and the overfit per-game
stores are sealed.  The genuinely GENERAL knowledge the substrate distilled must
still travel.  This script extracts every Tier-0 entry (the compliance scanner's
"compliant general" class -- no game ids, entity names, colours, or replay) from
the dev KB and writes a store-shaped, general-only copy into `kb_seed/`, which
`kaggle_entry.seed_general_kb()` copies into COS_KB_ROOT at startup.

The tier judgement is delegated to `kb_compliance_scan` (the single authority), so
this never re-implements the rules -- it only filters each store to the ids the
scanner deemed Tier-0 and re-wraps them in the store's own schema.

Run:  python usecases/arc-agi-3/submission/build_kb_seed.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "usecases/arc-agi-3/python"))
sys.path.insert(0, str(REPO / "tools/governor_audit/perception_loop_v2"))

import kb_compliance_scan as kcs   # noqa: E402

KB_ROOT = REPO / ".tmp/kb"
BENCH   = REPO / "usecases/arc-agi-3/benchmarks/knowledge_base"
SEED    = HERE / "kb_seed"


def _tier0_ids():
    """{store_name: set(entry_id)} for every Tier-0 entry, per the scanner."""
    res = kcs.scan(KB_ROOT, BENCH)
    recs = res[0] if isinstance(res, tuple) else res
    out: dict[str, set] = {}
    for r in recs:
        if r.tier() == 0:
            out.setdefault(r.store, set()).add(r.entry_id)
    return out


# --- per-store id schemes (mirror the scanner handlers EXACTLY) + re-wrap --------
def _gk(data, keep):
    items = data if isinstance(data, list) else (data.get("entries") or [])
    kept = [it for it in items if (it.get("id") or str(it.get("text", "?"))[:40]) in keep]
    return kept if isinstance(data, list) else {**data, "entries": kept}


def _err(data, keep):
    items = data if isinstance(data, list) else (data.get("errors") or [])
    kept = [e for i, e in enumerate(items) if e.get("category", f"err{i}") in keep]
    return kept if isinstance(data, list) else {**data, "errors": kept}


def _types(data, keep):
    return {**data, "types": [t for t in data.get("types", []) if t.get("type_id", "?") in keep]}


def _subs(data, keep):
    return {**data, "subroutines": [s for s in data.get("subroutines", [])
                                    if (s.get("subroutine_id") or s.get("name") or "?") in keep]}


def _tactics(data, keep):
    return {**data, "tactics": [t for t in data.get("tactics", [])
                                if (t.get("tactic_id") or t.get("name") or "?") in keep]}


def _xgame(data, keep):
    games = {}
    for gid, gobj in (data.get("games", {}) or {}).items():
        g = gid.split("-")[0]
        kk = [k for k in gobj.get("knowledge", []) if f"{g}:{k.get('type','?')}" in keep]
        if kk:
            games[gid] = {**gobj, "knowledge": kk}
    return {**data, "games": games}


FILTERS = {
    "general_knowledge.json":  _gk,
    "error_ledger.json":       _err,
    "task_type_library.json":  _types,
    "subroutine_kb.json":      _subs,
    "tactic_kb.json":          _tactics,
    "cross_game_knowledge.json": _xgame,
}


def _entry_texts(store, data):
    """Short human-readable lines for the manifest, per store schema."""
    out = []
    if store == "general_knowledge.json":
        for it in (data if isinstance(data, list) else data.get("entries", [])):
            out.append(f"[{it.get('kind','?')}] {str(it.get('text',''))[:140]}")
    elif store == "error_ledger.json":
        for e in (data if isinstance(data, list) else data.get("errors", [])):
            out.append(f"[{e.get('category','?')}] {str(e.get('description') or e.get('fix',''))[:130]}")
    elif store == "task_type_library.json":
        for t in data.get("types", []):
            out.append(f"[{t.get('type_id','?')}] {str(t.get('description',''))[:130]}")
    elif store == "subroutine_kb.json":
        for s in data.get("subroutines", []):
            out.append(f"[{s.get('subroutine_id') or s.get('name')}] {str(s.get('description',''))[:130]}")
    elif store == "tactic_kb.json":
        for t in data.get("tactics", []):
            out.append(f"[{t.get('tactic_id') or t.get('name')}] {str(t.get('intent') or t.get('name',''))[:130]}")
    elif store == "cross_game_knowledge.json":
        for gid, gobj in data.get("games", {}).items():
            for k in gobj.get("knowledge", []):
                out.append(f"[{k.get('type','?')}] {str(k.get('content',''))[:130]}")
    return out


def _instincts():
    import instincts as I
    insts = I.REGISTRY.all() if hasattr(I, "REGISTRY") and hasattr(I.REGISTRY, "all") \
        else [getattr(I, n) for n in dir(I) if isinstance(getattr(I, n), I.Instinct)]
    return [(getattr(i, "name", "?"), getattr(i, "category", "?"), str(getattr(i, "when", ""))[:90])
            for i in insts]


def main():
    SEED.mkdir(parents=True, exist_ok=True)
    tier0 = _tier0_ids()
    manifest = ["# Competition-mode knowledge (auto-generated by build_kb_seed.py)\n",
                "Everything the agent knows when it enters an UNSEEN game: code-resident "
                "instincts (always shipped) + the seeded general KB below.  NO per-game "
                "knowledge ships (claims, lessons, win-paths, visual memory are sealed).\n"]

    manifest.append("\n## A. Seeded general KB (Tier-0 only; copied into the fresh KB at start)\n")
    seeded = 0
    for store, fn in FILTERS.items():
        ids = tier0.get(store)
        src = KB_ROOT / store
        if not src.exists():
            src = BENCH / store          # some general stores live in the benchmark KB
        if not ids or not src.exists():
            continue
        data = json.loads(src.read_text(encoding="utf-8"))
        filtered = fn(data, ids)
        (SEED / store).write_text(json.dumps(filtered, indent=1, ensure_ascii=False), encoding="utf-8")
        lines = _entry_texts(store, filtered)
        seeded += len(lines)
        manifest.append(f"\n### {store}  ({len(lines)} entries)\n")
        manifest += [f"- {ln}" for ln in lines]

    manifest.append(f"\n\n## B. Code-resident instincts (ship in instincts.py, fire by trigger)\n")
    for name, cat, when in sorted(_instincts()):
        manifest.append(f"- **{name}** ({cat}) — _{when}_")

    (SEED / "COMPETITION_KNOWLEDGE.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"seed rebuilt: {seeded} general KB entries across {len([s for s in FILTERS if (SEED/s).exists()])} stores")
    print(f"manifest: {SEED / 'COMPETITION_KNOWLEDGE.md'}")


if __name__ == "__main__":
    main()
