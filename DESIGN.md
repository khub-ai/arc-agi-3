# arc-agi-3 — Design notes

This document records the boundary decisions for the ARC-AGI-3 domain
adapter.  The engine's own design spec lives in
[cognitive-os-engine/cognitive_os/DESIGN.md](https://github.com/khub-ai/cognitive-os-engine/blob/main/cognitive_os/DESIGN.md);
read that first — everything here is additive.

## 1. Standing invariants

These four rules are inherited from the engine's §14 standing invariants
and are non-negotiable:

1. **No game-specific code in the engine.** The engine depends on the
   adapter, never the reverse.  Every new reasoning capability must be
   a *generalisation of the substrate* (a new miner, claim type, or
   planner heuristic), not a branch on a game identifier.
2. **Every phase advances debugging, problem-solving, and tool
   creation.** These are the three capabilities the system exists to
   exercise.  Any phase that fails to advance all three is incomplete
   by definition.
3. **Cross-episode knowledge accumulation is first-class.** The
   `PostMortem`, `Option`, and `CachedSolution` channels must survive
   across episodes and (in future work) across games.
4. **The retired `ensemble.py` path does not return.** Accumulating
   per-game heuristics inside a single file is exactly the failure
   mode the extraction was designed to prevent.  Game-specific
   knowledge must live in learned artifacts (Options, cached
   solutions, hypothesis traces), not in code.

## 2. Package contract

```
ArcAdapter      ← cognitive_os.Adapter
   ├── reset / observe / execute / action_space / is_done      (SDK lifecycle)
   ├── invoke_tool                                             (dispatches to tools.registry)
   ├── observer_query  (Phase 5a: default "unsupported")       (Phase 5b: VLM)
   └── mediator_query  (Phase 5a: default "unsupported")       (Phase 5b: LLM)

perception     → build_observation(frame, state_name, levels, state) → Observation
action_mapping ⇄ engine_action_for / native_action_for
tools/         ← pure-Python grid primitives (bfs, components, symmetry, diff)
tools/registry → build_registry() : (ToolRegistry, handler_table)
```

The adapter has no other public surface.  Everything the engine sees is
synthesised from the `Observation` stream.

## 3. Observation schema

`Observation.agent_state` is the thin channel through which the adapter
surfaces terminal-state flags the engine can form goal conditions over.
Perception populates:

| Key                              | Type   | Meaning                                      |
|----------------------------------|--------|----------------------------------------------|
| `state_name`                     | str    | SDK state: `"PLAYING"` / `"WIN"` / `"GAME_OVER"` / … |
| `levels_completed`               | int    | raw SDK counter                              |
| `resources.levels_completed`     | float  | same, float-cast for `ResourceAbove`         |
| `resources.episode_won`          | float  | 1.0 on WIN, 0.0 otherwise                    |
| `resources.episode_lost`         | float  | 1.0 on GAME_OVER, 0.0 otherwise              |

The runner copies `agent_state` into `WorldState.agent` each step;
`ResourceAbove("episode_won", 0.5)` is the seed goal's condition.

## 4. Event emission rules

Perception emits at most one of each event type per frame:

| Event                    | Trigger |
|--------------------------|---------|
| `EntityAppeared`         | connected component present in this frame but not previous; key = (colour, normalised-shape) |
| `EntityDisappeared`      | previous-frame key absent in current frame |
| `EntityStateChanged`     | region with stable key translated; property `"centroid"` set to new centroid |
| `AgentDied`              | state transitioned to `GAME_OVER` |
| `GoalConditionMet`       | state transitioned to `WIN` (goal_id=`"episode"`) or mid-episode level-up (goal_id=`"level_N"`) |
| `SurpriseEvent`          | cells changed between frames without full motion-vector explanation |

Perception does not emit `AgentMoved` — the engine infers agent identity
from behavioural evidence (which region correlates with action
execution).  Emitting it at the adapter would hard-code which component
is the agent, violating invariant #1.

## 5. Tool suite

| Tool name                              | Purpose                                        |
|----------------------------------------|------------------------------------------------|
| `grid.bfs.shortest_path`               | 4-connected shortest path with passability fn |
| `grid.bfs.reachable_cells`             | BFS frontier                                   |
| `grid.components.label`                | connected-component labelling                  |
| `grid.components.extract_regions`      | labelled components with bbox / centroid / area |
| `grid.symmetry.detect`                 | horizontal / vertical / diagonal / 180°-rotational symmetry flags |
| `grid.diff.cell_diff`                  | per-cell change list                           |
| `grid.diff.motion_vectors`             | best-effort rigid-translation match per region |
| `grid.diff.is_identical`               | whole-frame equality (cheap)                   |

All are synchronous, deterministic, side-effect-free, and cheap (< 10 ms
on a 60×60 grid).  Async tools would be appropriate for a motion
planner; none are needed yet.

## 6. What is *not* here yet (Phase 5b)

* Observer backend (VLM) — the engine's default returns
  zero-confidence, which is the correct "no answer" signal.
* Mediator backend (LLM) — same as above.
* Harness CLI that connects to the live ARC-AGI-3 competition API.
* Recorded-frame replay fixtures from the live SDK (current tests use
  hand-built 5×5 episodes).
* Cross-episode `CachedSolution` persistence.

The LLM backends will be abstracted behind a small protocol with
Anthropic Claude as the first implementation.  An open-source LLM
(exact choice TBD) will be swapped in before the competition
submission — the protocol is designed to make that a single-file
change.

## 7. Test strategy

* `tests/test_tools.py` — per-tool unit tests on tiny fixtures.
* `tests/test_adapter.py` — Adapter ABC round-trip against a
  replay env.
* `tests/test_end_to_end.py` — `cognitive_os.run_episode` through
  the adapter on a synthetic WIN-terminating replay; the engine must
  mark the seed goal achieved and return `final_status="success"`.

CI never hits the live API.  Phase 5b adds recorded-fixture tests
from real games and gates them off the live harness by default.
