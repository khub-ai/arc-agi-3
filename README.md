# arc-agi-3

**ARC-AGI-3 domain adapter for the [Cognitive OS Engine](https://github.com/khub-ai/cognitive-os-engine).**

This repo is the in-progress [ARC Prize 2026](https://arcprize.org/competitions/2026) submission scaffold.  All reasoning lives in the engine (domain-agnostic, MIT-0); this repo contributes only the thin domain boundary:

| Layer | What it does |
|---|---|
| **`ArcAdapter`** | Translates `arc_agi` frames (60×60 palette grid, state, levels_completed, available_actions) into the engine's `Observation` + `Event`s + `EntityModel`s. Maps generic `Action` → `arc_agi.Action`. |
| **Tool suite** | BFS, connected-component labelling, symmetry detection, frame-diff / motion extractor — registered via `ToolRegistry` and invoked by miners, planner, and explorer. |
| **Observer** *(Phase 5b)* | Visual oracle: `frame + typed question → typed answer`, backed by a pluggable VLM (Claude at launch, open-source for competition). |
| **Mediator** *(Phase 5b)* | Common-sense oracle over `WorldStateSummary`, same pluggable backend surface. |
| **Harness** *(Phase 5b)* | CLI that connects to the ARC-AGI-3 competition API and runs episodes through `cognitive_os.run_episode`. |

> **License:** [MIT No Attribution (MIT-0)](LICENSE) — the ARC Prize competition rules require permissive public-domain-equivalent licensing (CC0 or MIT-0). This repo and its engine dependency both comply.

## Install

```bash
pip install -e .          # adapter + domain tools only
pip install -e ".[llm]"   # + Anthropic SDK (Observer/Mediator backends)
pip install -e ".[dev]"   # + pytest
```

## Status

| Phase | Deliverable | Status |
|---|---|---|
| 5a | Scaffold + tool suite + adapter skeleton (no LLM) | Done |
| 5b | Observer + Mediator + harness + full episode run | Done |
| 5c | Cross-episode knowledge persistence (CachedSolutions) | Done |
| 6a | Live SDK parity (dry-run, real arc_agi types verified) | Done |
| 6b | Open-source LLM backend | Pending |
| 6c | Competition dry run | Pending |

## Architecture invariants

These mirror the engine's [standing directives](https://github.com/khub-ai/cognitive-os-engine/blob/main/cognitive_os/DESIGN.md) — they are not negotiable:

1. **No game-specific code leaks into the engine.** Every new capability is a generalisation of the substrate (new miner, claim type, planner heuristic), not a branch on a game identifier.
2. **Every phase advances debugging, problem-solving, and tool creation.** These three capabilities are the reason the system exists.
3. **Cross-episode knowledge accumulation is first-class.** `PostMortem`, `Option`s, and `CachedSolution`s survive across games.
4. **Never re-import the retired `ensemble.py` game-specific heuristics.** The replaced system accumulated per-game logic that bloated and stopped generalising; that path does not come back.

## Running an episode

```bash
export ARC_API_KEY=...                # competition API key
export ANTHROPIC_API_KEY=...          # only for --backend anthropic

arc-agi-3 --game-id ls20 --dry-run                         # 10-step live parity check
arc-agi-3 --game-id ls20                                   # null backend, ephemeral
arc-agi-3 --game-id ls20 --backend anthropic               # Claude Observer/Mediator
arc-agi-3 --game-id ls20 --episodes 5                      # intra-run accumulation
arc-agi-3 --game-id ls20 --knowledge-dir ./store           # persist across invocations
arc-agi-3 --game-id ls20 --knowledge-dir ./store \
          --no-save-knowledge                              # read-only (competition)
```

The `--dry-run` flag runs a bounded 10-step episode with `NullBackend`
(zero LLM cost) and prints a parity report — frame shape, state-name
transitions, action space size, entity / hypothesis / surprise
counts.  Run it first after any SDK upgrade to confirm adapter
shapes still agree with whatever the live API hands back.

## Tests

```bash
pytest
```

Tests use synthetic / recorded fixtures only; the live ARC-AGI-3 API is not hit in CI.
