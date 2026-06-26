"""Kaggle entry config + agent factory.

What Kaggle actually runs is determined by the competition's starter notebook /
harness (confirm against it - see README "To verify"). This module is the one
place we (1) force the OFFLINE + strict-mode config and (2) hand back the agent
to register with the framework, so the notebook stays a thin shell:

    from kaggle_entry import enforce_offline, build_agent
    enforce_offline()
    agent = build_agent("cos")        # or "baseline" for the Day 1-2 shakedown
    # ... register/run `agent` via the framework harness ...
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def seed_general_kb() -> int:
    """Seed the fresh competition KB with GAME-AGNOSTIC priors only.

    The scratch KB starts empty (no overfit per-game knowledge -- that is sealed in
    strict mode).  But the substrate DID distil genuinely general priors from earlier
    play (`general_knowledge.json`, all Tier-0 in the compliance scanner: no game ids,
    no entity names, no colours).  Those are competition-legal and worth carrying, so
    copy the shipped `kb_seed/` stores into COS_KB_ROOT.  kb_recall reads
    `general_knowledge` (via `load_general`) and surfaces these situationally.

    Idempotent: a store already present in the scratch KB is left untouched, so a
    restart preserves anything learned live this session.  Returns # stores seeded."""
    root = Path(os.environ.get("COS_KB_ROOT", "/kaggle/working/kb"))
    seed_dir = Path(__file__).resolve().parent / "kb_seed"
    if not seed_dir.is_dir():
        return 0
    root.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(seed_dir.glob("*.json")):
        dst = root / src.name
        if not dst.exists():
            shutil.copy2(src, dst); n += 1
    return n


def enforce_offline() -> None:
    """Hard-set the competition-clean config: no network, strict COS, local
    model only. Call FIRST, before importing model/agent code."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["NO_PROXY"] = "*"
    # COS: strict mode (dev_only sealed, no overfit/replay KB), no API backends.
    os.environ["COS_STRICT"] = "1"
    os.environ.setdefault("COS_DISABLE_API_BACKENDS", "1")
    # Point COS's KB at a clean, writable scratch dir (NOT a baked overfit KB).
    os.environ.setdefault("COS_KB_ROOT", "/kaggle/working/kb")
    # Seed ONLY the game-agnostic priors into that scratch KB (universals already
    # ship as instincts in code; this carries the conditional general strategies).
    seed_general_kb()


def build_agent(kind: str = "cos", **agent_kwargs):
    """Return an Agent instance. `agent_kwargs` are the framework's
    Agent.__init__ args (card_id, game_id, agent_name, ROOT_URL, record,
    arc_env, tags) supplied by the harness."""
    if kind == "baseline":
        from baseline_agent import BaselineAgent
        return BaselineAgent(**agent_kwargs)
    if kind == "cos":
        from cos_agent import CosAgent
        return CosAgent(**agent_kwargs)
    raise ValueError(f"unknown agent kind {kind!r}")


if __name__ == "__main__":
    # Smoke check: config + importability only (no harness here).
    enforce_offline()
    import _compat
    print("framework importable:", _compat.FRAMEWORK_AVAILABLE)
    print("offline env set:", os.environ.get("TRANSFORMERS_OFFLINE"),
          "| strict:", os.environ.get("COS_STRICT"))
