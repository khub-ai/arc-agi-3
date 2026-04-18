# Dialogic Distillation — Observer / Mediator Spec

*Revision 2 — review-driven rewrite (2026-04-18).  Key changes from
r1: metadata captured at cache-write-time rather than via a post-hoc
wallclock scanner; scene keys and heuristics rank-normalised so
colour IDs don't leak across games; validation filter gates entry
inclusion; tutor-probe script decouples corpus growth from
level-solving; primary eval gate moved from `role_iou_vs_tutor` to
ground-truth `episode_win_rate` + `role_iou_vs_ground_truth`; dedup
and skipped-entry accounting in the KF build; three-arm eval
(Tutor / PUPIL-raw / PUPIL-DD) replaces the two-arm protocol.*

## Purpose

The ARC-AGI-3 competition submission must run end-to-end under an
open-source LLM, but current development uses Sonnet 4.6 for the
Observer and Mediator roles.  This document specifies how to use
**Dialogic Distillation (DD)** to bring an OSS model (Qwen-2.5-72B
or Llama-3.3-70B) up to Sonnet-comparable quality on these two
narrow roles without fine-tuning.

The roles are narrow on purpose: the engine never asks the LLM to
reason generally.  It asks typed, JSON-schema-constrained questions.
That narrow surface is why DD works here.

---

## How DD works (one-paragraph summary)

DD is a retrieval-augmented inference protocol, not training.

1. **Tutor corpus** — run Sonnet on N games × K levels.  Every
   prompt+reply is already captured to `.tmp/oracle_cache/*.json` by
   `CachedChatBackend`.  No new infrastructure required to collect it.

2. **Knowledge Fabric (KF)** — a post-processing step converts the
   cache into abstracted KF entries keyed on *scene features*, not
   raw frames.  Two separate pools: `observer_kf.jsonl` and
   `mediator_kf.jsonl`.

3. **DD inference** — a `DDChatBackend` wraps the OSS (PUPIL) backend.
   On each `chat()` call it extracts scene features from the prompt,
   retrieves the top-K matching KF entries, and injects them into the
   user message before forwarding to the PUPIL.  No change to the
   engine, triggers, observer, or mediator modules.

---

## Phase gate

**Do not start DD experiments until:**

- [ ] At least one full Sonnet run on ls20 L1 produces a non-empty
      oracle cache (a single `ENUMERATE_OBJECTS` answer counts).
      *Current cache at `.tmp/oracle_cache/` already has one entry.*
- [ ] Cache-metadata capture landed (see §1.2) — without it the
      corpus cannot be keyed or validated.
- [ ] Tutor-probe script landed (see §1.3) — without it corpus
      growth is bottlenecked on solving games we haven't solved yet.
- [ ] At least **two additional games** (e.g. vc33, sp80) have been
      run with Sonnet + cache so the KF is not overfit to ls20's
      palette.  Sonnet must clear a smoke pass on each held-out game
      (i.e. the adapter can reset and the initial frame parses); if
      it doesn't, DD has no tutor answer to match.
- [x] GAP 7 (Mediator causal linkage) is wired — so
      `PROPOSE_GOAL_LINKAGE` queries fire and produce cache entries
      to distil from.  Landed in engine `fe8feac` / arc-agi-3
      `d83967e`.

The cache already works; the blockers are (a) metadata capture so
entries are retrievable, (b) diversity of tutor examples without
needing to solve every game first, and (c) a validation filter so we
don't distil tutor mistakes.

---

## Part 1 — Tutor corpus (what the COS session must collect)

### 1.1  Existing cache format (already sufficient for corpus capture)

`CachedChatBackend` writes one JSON file per unique call:

```json
{
    "key":         "<sha1>",
    "model":       "claude-sonnet-4-6",
    "max_tokens":  2000,
    "temperature": 0.0,
    "messages":    [
        {"role": "system", "content": "..."},
        {"role": "user",   "content": "QUESTION_TYPE: ENUMERATE_OBJECTS\n..."}
    ],
    "reply":       "{\"result\": [...], \"confidence\": 0.85, ...}",
    "created_at":  1745001600.0,
    "latency_ms":  14300.0
}
```

**The cache IS the tutor corpus.**  No new backend wrapper is needed.

### 1.2  Metadata capture — at write-time, not post-hoc

**The sidecar-scanner approach was rejected** (prior spec revision).
Post-hoc scanning on a wallclock cutoff is racy (long runs exceed
the window), not idempotent (rerunning on older caches orphans
every entry), and loses data on crash.  Metadata is a property of
the call, not of the file's mtime.

**Correct design: extend `CachedChatBackend` to carry call-scoped
metadata and merge it into the stored JSON at write-time.**

```python
class CachedChatBackend(ChatBackend):
    # NEW: call-scoped metadata, set by the caller before each chat()
    metadata: dict = field(default_factory=dict)

    def chat(self, messages, *, max_tokens=1024, temperature=0.0) -> str:
        key = _hash(messages, max_tokens, temperature, self._model_id)
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())["reply"]   # HIT
        # MISS — forward to inner, persist with current metadata
        reply = self._inner.chat(messages, max_tokens=max_tokens,
                                 temperature=temperature)
        path.write_text(json.dumps({
            "key":         key,
            "model":       self._model_id,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "messages":    [asdict(m) for m in messages],
            "reply":       reply,
            "created_at":  time.time(),
            "metadata":    dict(self.metadata),   # merged in at write-time
        }, ensure_ascii=False))
        return reply
```

Only *misses* write — hits don't need re-tagging because they were
tagged on their original write.  The loop runner sets
`backend.metadata` before each `run_episode` (and per-step if the
step is interesting for metadata) and is done.

**Required metadata fields** (COS session must populate all of these
before invoking `run_episode`):

| Field | Why |
|---|---|
| `game_id`, `level` | Corpus sharding and cross-game transfer analysis |
| `episode_id`, `attempt_idx`, `step` | Same game+level produces different answers at different points in an episode; required to sequence retrieval |
| `engine_sha`, `arcagi3_sha` | Forensics for cache-entry provenance when prompts evolve |
| `prompt_sha` | SHA-1 of the prompt-template string used by the trigger / question builder.  Entries from an obsolete prompt shape must be auto-invalidated at retrieval time |
| `ws_features` | Engine-observed features at call time: `entity_count`, `committed_role_set`, `agent_position_known`, `n_committed_causals_matching_episode_won`.  Saves the KF build script from re-segmenting frames in §3 |
| `episode_outcome` | Populated at episode end (`win` / `loss` / `timeout`) via a post-run patch of every cache entry written during the episode — this is the only field that CAN'T be set at write-time, because the outcome isn't yet known.  Loop runner walks cache entries with `metadata.episode_id == <just-finished>` and sets `metadata.episode_outcome` once.  Used by the validation filter (§3) |

**Why `episode_outcome` is a permitted post-hoc update**: unlike
`game_id`, it genuinely isn't knowable at call time.  The loop
runner patches it exactly once per episode, keyed on `episode_id`
rather than wallclock — idempotent and crash-safe (a cached entry
whose episode never terminated gets `episode_outcome="crashed"`
which is meaningful too).

**Target corpus size before starting DD eval:**
   - `ENUMERATE_OBJECTS`: ≥15 entries across ≥5 games
   - `PROPOSE_GOAL_LINKAGE`: ≥10 entries across ≥5 games (now that
     GAP 7 is wired, these accrue on every level-start)
   - Other Mediator question types: 0 is acceptable for the first
     DD eval pass; add once `EXPLAIN_SURPRISE` / `SUGGEST_MECHANICS`
     triggers land

### 1.3  Tutor-probe script — decouple corpus from level-solving

**The chicken-and-egg problem.**  `ENUMERATE_OBJECTS` fires once
per level.  `PROPOSE_GOAL_LINKAGE` fires once per level.  Reaching
level 2 requires the COS to solve level 1.  We haven't solved level
1 yet.  Therefore the "run Sonnet on ≥5 games × several levels"
target is unreachable through full `run_episode` alone.

**Fix: a probe-only harness** that instantiates adapter + world
state for a game, advances far enough for `InitialFrameScanTrigger`
and `EpisodeGoalLinkageTrigger` to fire exactly once each, saves
the cache entries with full metadata, and exits — no planner, no
multi-step episode, no need to win.

```python
# .tmp/tutor_probe.py
for game_id in ["ls20", "vc33", "sp80", "ft09", "ls33"]:
    env     = arcade.make(game_id)
    adapter = ArcAdapter(raw_env=env, env_id=game_id, backend=backend)
    ws      = WorldState()
    cfg     = EngineConfig.arc_agi3_default()
    backend.metadata = {
        "game_id":     game_id,
        "level":       1,
        "episode_id":  f"probe::{game_id}::L1",
        "attempt_idx": 0,
        "step":        0,
        "engine_sha":  _git_head("cognitive-os-engine"),
        "arcagi3_sha": _git_head("arc-agi-3"),
    }
    # Drive the engine just enough to fire both triggers.  max_steps=2
    # is enough: step 1 ingests the initial frame (fires
    # InitialFrameScanTrigger); step 2 evaluates goals (fires
    # EpisodeGoalLinkageTrigger via the oracle dispatch pass).
    run_episode(adapter, ws, cfg, episode_id=backend.metadata["episode_id"],
                max_steps=2)
    _patch_episode_outcome(cache_dir, backend.metadata["episode_id"],
                            outcome="probe")
```

Runtime: ~5 seconds per game (one live API call miss each if
uncached).  Produces `2 × N_games` corpus entries per pass.  This
is the primary corpus-growth mechanism until L1 is solved on
several games.

---

## Part 2 — Abstract scene key (grounding for retrieval)

Frames are unique per game, so KF entries cannot be keyed on raw
frames.  They are keyed on **abstract visual features** computed from
the frame.  These are cheap to derive from the JSON grid already in
the prompt — and, when `ws_features` metadata is populated at call
time (§1.2), many of them don't need re-derivation in the build step
at all.

### 2.1  Observer scene features (for `ENUMERATE_OBJECTS`)

**Guiding principle: colour IDs do not transfer between games.**
Colour 9 is the ls20 agent but could be a vc33 hazard.  Keying on
literal colour IDs and then distilling "colour 9 = agent" poisons
cross-game retrieval.  So the scene key and the heuristic text are
both expressed in **rank-normalised** palette terms: the dominant
(background) colour is rank 0, the next-most-frequent is rank 1,
and so on.  A ls20 entry saying "rank-3 isolated sprite = agent"
retrieves correctly for a vc33 frame where the rank-3 colour
happens to be a different literal ID.

```python
@dataclass
class ObserverSceneKey:
    grid_rows:              int          # e.g. 64
    grid_cols:              int          # e.g. 64
    palette_size:           int          # len(distinct colours), e.g. 5
    # Rank-normalised palette, by cell-count descending:
    # palette_ranks[i] = literal colour whose rank is i (rank 0 = background).
    # Stored but NOT used as a similarity key — only for traceability.
    palette_ranks:          list[int]    # e.g. [4, 3, 5, 9, 12]
    # Relative frequencies at each rank, bucketed for stability.
    palette_freq_buckets:   list[str]    # e.g. ["dominant", "large", "medium", "small", "small"]
    object_count_bucket:    str          # "1-3", "4-6", "7-10", "11+"
    # prompt_sha and ws_features from metadata (§1.2) are additional
    # filter keys applied before similarity — not part of the scored
    # similarity itself.
```

**Heuristic texts are rewritten to use ranks, not literal colours.**
Templates in §3.3 produce strings like "isolated sprite of rank-3
colour on rank-0 background = agent", never "colour 9 = agent".  A
build-step guard rejects any heuristic whose text contains a literal
colour ID numeric.

**Similarity for retrieval (weights are v1 placeholders; eval-tuned):**

```python
def scene_similarity(a: ObserverSceneKey, b: ObserverSceneKey) -> float:
    # Palette-size proximity: exact = 1.0, off-by-one = 0.7, else 0.3
    size_match = {0: 1.0, 1: 0.7}.get(abs(a.palette_size - b.palette_size), 0.3)
    # Frequency-bucket overlap: how many ranks have the same size-bucket
    freq_overlap = sum(
        1 for fa, fb in zip(a.palette_freq_buckets, b.palette_freq_buckets)
        if fa == fb
    ) / max(len(a.palette_freq_buckets), len(b.palette_freq_buckets), 1)
    bucket_match = 1.0 if a.object_count_bucket == b.object_count_bucket else 0.3
    return size_match * 0.35 + freq_overlap * 0.35 + bucket_match * 0.30
```

Weights 0.35 / 0.35 / 0.30 are placeholder.  First eval pass reports
sensitivity to these; tune with a small grid search against
`role_iou_vs_ground_truth` (§5).

### 2.2  Mediator scene features

The Mediator receives a `WorldStateSummary`, not raw frames.  The
scene key is derived from the summary JSON already in the user
message, plus the `ws_features` block captured in metadata (§1.2).

```python
@dataclass
class MediatorSceneKey:
    question_type:        str          # primary shard key
    entity_count_bucket:  str          # "1-3", "4-6", "7-10", "11+"
    committed_role_set:   list[str]    # sorted role values from committed PropertyClaims
    # For bootstrap questions fired before any role commits:
    proposed_role_set:    list[str]    # roles from PROPOSED (contested) role claims
    has_causal_toward_episode_won: bool
```

Retrieval: exact match on `question_type`, then sort by
`entity_count_bucket` match + `committed_role_set` overlap +
`proposed_role_set` overlap.

**Known weakness — PROPOSE_GOAL_LINKAGE.**  GAP 7's trigger fires
specifically *before* any causal claim matching episode_won is
committed (that's its whole precondition).  So the
`committed_role_set` and `has_causal_toward_episode_won` fields are
maximally uninformative for exactly the question type that most
needs them.  Fallback ordering is `question_type` →
`entity_count_bucket` → `proposed_role_set`.  Accept that this will
be the lowest-precision retrieval of any question type in v1; if
eval shows it underperforming, next step is to add colour-histogram
features or a structural-entity fingerprint (top-3 entity
colour/size signatures).

---

## Part 3 — KF build script

**Path:** `arc_agi_3/tools/build_kf.py`

```
Usage: python -m arc_agi_3.tools.build_kf \
           --cache-dir .tmp/oracle_cache \
           --output-dir .tmp/kf \
           [--validate-only]          # drop entries with validated=False
           [--current-prompt-shas sha1,sha2,...]   # invalidate stale prompt shapes
```

Outputs:
- `.tmp/kf/observer_kf.jsonl`   — one entry per ENUMERATE_OBJECTS tutor answer
- `.tmp/kf/mediator_kf.jsonl`   — one entry per Mediator tutor answer
- `.tmp/kf/build_report.json`   — skipped counts by reason (see §3.4)

### 3.1  KF entry schema — Observer

```json
{
    "entry_id":        "obs_ls20_a3f2c1",
    "question_type":   "ENUMERATE_OBJECTS",
    "game_id":         "ls20",
    "level":           1,
    "episode_id":      "ls20::l1_loop::attempt_1",
    "prompt_sha":      "b72e10...",
    "scene_key": {
        "grid_rows":            64,
        "grid_cols":            64,
        "palette_size":         5,
        "palette_ranks":        [4, 3, 5, 9, 12],
        "palette_freq_buckets": ["dominant", "large", "medium", "small", "small"],
        "object_count_bucket":  "7-10"
    },
    "tutor_answer": {
        "objects": [
            {"role": "agent",  "colour_rank": 3,  "shape": "small_sprite",
             "bbox": [11,35,13,37], "description": "3x3 moving sprite"},
            {"role": "target", "colour_rank": 1,  "shape": "bordered_box",
             "bbox": [32,8,40,16], "description": "9x9 bordered box"},
            {"role": "wall",   "colour_rank": 1,  "shape": "line",
             "bbox": [0,0,64,3],  "description": "top border"}
        ],
        "confidence": 0.85,
        "explanation": "grid world with navigable space and bordered target box"
    },
    "heuristics": [
        "An isolated sprite of rank-3 colour on rank-0 background is the agent",
        "A bordered box of the same colour as the walls with interior space is a target"
    ],
    "validated":         false,
    "validation_reason": null,
    "tutor_model":       "claude-sonnet-4-6",
    "created_at":        1745001600.0
}
```

Literal colour IDs are replaced by ranks in both `tutor_answer` and
`heuristics`.  The rank→literal mapping lives in
`scene_key.palette_ranks` for forensic reconstruction.

### 3.2  KF entry schema — Mediator

```json
{
    "entry_id":      "med_ls20_b1e9d4",
    "question_type": "PROPOSE_GOAL_LINKAGE",
    "game_id":       "ls20",
    "level":         1,
    "episode_id":    "probe::ls20::L1",
    "prompt_sha":    "c1a4e3...",
    "scene_key": {
        "question_type":                 "PROPOSE_GOAL_LINKAGE",
        "entity_count_bucket":           "7-10",
        "committed_role_set":            [],
        "proposed_role_set":             ["agent", "target", "wall"],
        "has_causal_toward_episode_won": false
    },
    "tutor_answer": {
        "causal_links": [
            {"trigger_kind": "AtPosition", "effect_kind": "ResourceAbove",
             "effect_resource_id": "episode_won"}
        ],
        "confidence":  0.28,
        "explanation": "scan identified a target entity; reaching it is the plausible win trigger"
    },
    "heuristics": [
        "When the only leaf of the episode goal is ResourceAbove(episode_won), propose a CausalClaim(AtPosition(agent, target_centroid) -> ResourceAbove(episode_won))",
        "When a scan_* entity has kind=target, prefer its centroid as the trigger position"
    ],
    "validated":         false,
    "validation_reason": null,
    "tutor_model": "claude-sonnet-4-6",
    "created_at":  1745001600.0
}
```

### 3.3  Validation filter — the load-bearing safeguard

**Tutor is not oracle.**  Observed on ls20 L1: Sonnet reported agent
`colour="0"` (wrong), agent position `[21,30]` (off by ~10 px),
target at `[34,43]`/`[36,43]` (actually `scan_06`, which it separately
labelled "resource").  If the build script distils every cache entry
verbatim, those errors become permanent retrieval priors for every
pupil.  Therefore every KF entry carries a `validated` flag.

**Validation sources (any one flips `validated=True`):**

1. **Symbolic confirmation.**  The engine later committed a claim
   compatible with the tutor's assertion.  Example: tutor says
   "agent is colour-rank 3 isolated sprite"; engine subsequently
   committed `ControlledActorClaim(colour=9, background=4)` whose
   `colour` has rank 3 in the scene's palette.  Confirmation ⇒
   validated.
2. **Outcome confirmation.**  An episode used the tutor's answer
   (via the live retrieval path) and terminated in `win`.
   `metadata.episode_outcome == "win"` on the generating cache
   entry is sufficient.
3. **Peer confirmation (cross-game).**  Two independent tutor calls
   on different `game_id`s both produced the same heuristic text
   after rank-normalisation.

`validation_reason` records which source fired (`"symbolic:ControlledActorClaim"`,
`"outcome:win:ls20::l1_loop::attempt_5"`, `"peer:med_ls33_e99f12"`).

`--validate-only` drops unvalidated entries from the output JSONL.
Default is to INCLUDE all entries but flag them; the DD backend
then retrieves only `validated=True` by default (§4.3).

### 3.4  Dedup, invalidation, and skipped-entry accounting

**Dedup.**  Multiple runs of the same game produce near-identical
ENUMERATE_OBJECTS entries.  Without dedup the KF becomes dominated
by whichever game was run most.  Build-time dedup key:

```python
dedup_key = (question_type, game_id, level, prompt_sha,
             scene_key.palette_ranks_signature())
```

When duplicates exist, keep the most recent validated entry; else
the most recent entry overall.  Build report records
`duplicates_collapsed` counts.

**Invalidation by prompt evolution.**  If the CLI flag
`--current-prompt-shas` is provided, entries whose `prompt_sha` is
not in the allowed set are dropped with reason
`"stale_prompt_sha"`.  Without the flag, all shas are accepted
(useful for retrospective builds).

**Skipped-entry accounting (`build_report.json`):**

```json
{
    "entries_scanned":      312,
    "entries_kept":         47,
    "skipped": {
        "malformed_cache_file":       3,
        "missing_metadata":           8,
        "reply_not_parseable":        12,
        "unknown_question_type":      4,
        "literal_colour_in_heuristic": 9,
        "duplicates_collapsed":       229
    },
    "validated": {"true": 14, "false": 33},
    "by_question_type": {"ENUMERATE_OBJECTS": 31, "PROPOSE_GOAL_LINKAGE": 16},
    "by_game_id":       {"ls20": 22, "vc33": 14, "sp80": 11}
}
```

The `literal_colour_in_heuristic` counter catches heuristic texts
that slipped through with hard-coded colour IDs — build MUST
reject them, because they would defeat rank-normalised retrieval
(§2.1 guiding principle).

### 3.5  Build script logic

```python
for cache_file in cache_dir.glob("*.json"):
    try:
        entry = json.loads(cache_file.read_text())
    except Exception:
        report["skipped"]["malformed_cache_file"] += 1
        continue
    if "metadata" not in entry or "game_id" not in entry["metadata"]:
        report["skipped"]["missing_metadata"] += 1
        continue
    q_type = _extract_question_type(entry["messages"])
    try:
        reply_parsed = json.loads(entry["reply"])
    except Exception:
        report["skipped"]["reply_not_parseable"] += 1
        continue
    if q_type == "ENUMERATE_OBJECTS":
        scene_key  = _compute_observer_scene_key_from_metadata(entry["metadata"])
        objects    = reply_parsed["result"]
        heuristics = _derive_heuristics(objects, scene_key)  # ranks only
        if _any_contain_literal_colour(heuristics):
            report["skipped"]["literal_colour_in_heuristic"] += 1
            continue
        kf_entry = build_observer_entry(scene_key, objects, heuristics, entry)
        _maybe_validate(kf_entry, entry["metadata"], symbolic_index,
                         peer_index)
        observer_kf.append(kf_entry)
    elif q_type in MEDIATOR_QUESTION_TYPES:
        ...
```

**Heuristic templates — all rank-expressed, no literal colours:**

```python
_HEURISTIC_TEMPLATES = {
    "agent":    "An isolated {shape} of rank-{colour_rank} colour on rank-0 background is the agent",
    "target":   "A {shape} of rank-{colour_rank} colour bounded within the playfield is a target",
    "wall":     "A large {shape} of rank-{colour_rank} colour covering the frame border is a wall",
    "hazard":   "A {shape} of rank-{colour_rank} colour that appears after bad outcomes is a hazard",
    "resource": "A {shape} of rank-{colour_rank} colour that disappears on contact is a resource",
    "decor":    "A {shape} of rank-{colour_rank} colour near the frame edge is HUD/score, treat as decor",
}
```

**Known v2 item: negative examples.**  Observed contradictions of
tutor claims ("scene had rank-2 HUD; tutor said HUD was rank-3 —
use caution") would be as useful as positive examples.  Schema
supports this via a `polarity: "positive" | "negative"` field per
heuristic; v1 emits only positives, v2 adds negatives once the eval
loop tags contradicted entries.

---

## Part 4 — DD backend

**Path:** `arc_agi_3/backends/dd_backend.py`

The `DDChatBackend` is the PUPIL wrapper.  It is a drop-in replacement
for `AnthropicBackend` from the engine's perspective.

```python
class DDChatBackend(ChatBackend):
    """
    Wraps a PUPIL ChatBackend with KF retrieval injection.

    On each chat() call:
      1. Extract question_type and scene features from messages.
      2. Retrieve top-K KF entries by scene similarity.
      3. Inject entries as a RETRIEVED_KNOWLEDGE block into the
         user message, between DESCRIPTION and FRAMES.
      4. Forward augmented messages to pupil.chat().
    """

    def __init__(
        self,
        pupil:             ChatBackend,
        *,
        observer_kf:       Path,   # path to observer_kf.jsonl
        mediator_kf:       Path,   # path to mediator_kf.jsonl
        top_k:             int  = 4,
        per_game_cap:      int  = 2,   # max entries from any one game_id in top_k
        validated_only:    bool = True,  # see §3.3
        min_similarity:    float = 0.4,  # below this, treat as retrieval miss
    ) -> None: ...

    def chat(
        self,
        messages:    list[ChatMessage],
        *,
        max_tokens:  int   = 1024,
        temperature: float = 0.0,
    ) -> str:
        q_type  = _extract_question_type(messages)
        entries = self._retrieve(q_type, messages)
        # Retrieval miss: silently fall through unaugmented.  This is
        # the right behaviour — we'd rather give the pupil no hint
        # than a misleading one, and an unaugmented pupil answer is
        # itself data worth capturing.
        if entries:
            messages = _inject_kf(messages, entries)
        return self.pupil.chat(messages, max_tokens=max_tokens,
                               temperature=temperature)
```

**Per-game cap.**  With target corpus ~15 entries / question type,
naive top-4 retrieval can pull all 4 from the most-represented
game.  `per_game_cap=2` forces diversity: after ranking, at most
`per_game_cap` entries per `game_id` survive into the top-k slot.
This matters most before the corpus is balanced across games.

**Validated-only default.**  §3.3 establishes that `validated=True`
entries are safer priors than unvalidated ones.  Default behaviour:
retrieve only validated entries, fall back to unvalidated only if
fewer than 2 validated entries pass `min_similarity`.  An eval-time
flag `validated_only=False` disables the filter for ablations.

**`min_similarity` cutoff.**  Without a floor, `_retrieve` returns
the top-k by rank even if the best match scored 0.1.  A floor of 0.4
means "if no entry is at least somewhat similar, don't inject
anything" — silent fallback to the unaugmented pupil.  This is the
correct behaviour on truly novel scenes; the wrong one would be to
inject a barely-matching entry and mislead.

### 4.1  KF injection format (into user message)

Insert a `RETRIEVED_KNOWLEDGE:` block after `DESCRIPTION:` and before
`FRAMES:` (Observer) or `WORLD_SUMMARY:` (Mediator):

```
QUESTION_TYPE: ENUMERATE_OBJECTS

DESCRIPTION: Enumerate every distinct visible object...

RETRIEVED_KNOWLEDGE:
The following heuristics were validated by a more capable model on
similar scenes.  Use them as strong priors when answering:

[Entry 1 — palette=[3,4,5,9], 7-10 objects]
- Small isolated sprite of colour 9 on background 4 = agent
- Bordered box of wall-colour with interior space = target/goal
- Colour-5 irregular region near frame edge = HUD/score, treat as decor

[Entry 2 — palette=[3,4,9,12], 4-6 objects]
- Two-colour sprite {9,12} is a multi-tile agent
- Single-colour bordered box = the sole target

FRAMES:
FRAME[0]:
[[4,4,4,...],...]
```

Key: inject *heuristics*, not raw example answers.  Heuristics
transfer across frames; raw answers do not.

### 4.2  PUPIL backend (first target)

```python
# arc_agi_3/backends/ollama_backend.py  (or vllm_backend.py)
class OllamaBackend(ChatBackend):
    """
    ChatBackend wrapping a locally-served Qwen-2.5-72B-Instruct via
    Ollama or vLLM.  Implements the same chat() contract.
    """
    def __init__(self, *, base_url: str, model: str, **kwargs): ...
    def chat(self, messages, *, max_tokens=1024, temperature=0.0) -> str: ...
```

The `DDChatBackend` wraps this:

```python
pupil   = OllamaBackend(base_url="http://localhost:11434", model="qwen2.5:72b")
dd      = DDChatBackend(pupil, observer_kf=kf_dir/"observer_kf.jsonl",
                                mediator_kf=kf_dir/"mediator_kf.jsonl")
adapter = ArcAgi3Adapter(game_id="ls20", backend=dd, ...)
```

No other change to the engine or adapter is needed.

---

## Part 5 — Eval protocol

**Prior spec revision used `role_iou_vs_tutor ≥ 0.80` as the
primary gate.  That's self-referential: it measures "does PUPIL
reproduce tutor's mistakes" rather than "does PUPIL get it right."**
Tutor is not oracle (§3.3).  So the eval gate is now rooted in
ground-truth outcomes where available, and agreement-with-tutor is
demoted to diagnostic.

### 5.1  Three parallel sessions on the same held-out games

Need a THIRD arm to separate DD lift from pupil baseline:

| Session | Backend | Purpose |
|---|---|---|
| Tutor (ceiling) | `CachedChatBackend(AnthropicBackend(...))` | Baseline ceiling |
| PUPIL-raw (floor) | `CachedChatBackend(OllamaBackend(...))` | Baseline floor — pupil with no DD |
| PUPIL-DD | `CachedChatBackend(DDChatBackend(OllamaBackend(...)))` | Does the KF actually help? |

Without the PUPIL-raw arm, we can't tell if PUPIL-DD beating
PUPIL-raw is due to the KF or due to noise.  Separate cache dirs
per session so runs don't cross-contaminate.

### 5.2  Metrics — grouped by ground-truth proximity

**Tier 1 — ground-truth gates (promoted, primary):**

| Metric | Description | Target |
|---|---|---|
| `episode_win_rate` | % episodes the full COS loop wins on held-out games | PUPIL-DD ≥ Tutor − 15% |
| `role_iou_vs_ground_truth` | IoU of role labels vs adapter-provided labels where available (agent identity via `ControlledActorClaim`, target position via win-cell replay) | PUPIL-DD ≥ 0.60 |
| `parse_ok` | % PUPIL replies that parse to valid JSON | ≥ 95% |

**Tier 2 — tutor-agreement diagnostics (demoted, report only):**

| Metric | Description | Use |
|---|---|---|
| `role_iou_vs_tutor` | IoU of role labels vs tutor | Sanity check; a PUPIL-DD score far above PUPIL-raw on this metric without a corresponding Tier-1 lift indicates "pupil reproduces tutor mistakes more faithfully" and should be treated as suspicious |
| `colour_agree_vs_tutor` | % roles where PUPIL colour-rank matches tutor | Diagnostic only |
| `confidence_mean` | Mean PUPIL confidence | Sanity |

**Primary gate (Tier 1):**
1. `episode_win_rate` on PUPIL-DD within 15pp of Tutor's win rate
   on the same games AND
2. `role_iou_vs_ground_truth` ≥ 0.60 on `ENUMERATE_OBJECTS`.

If Tier-1 gates pass but Tier-2 `role_iou_vs_tutor` is
embarrassingly low, that's fine — PUPIL-DD agreed with the world,
not with the tutor, and the world is what we're graded on.

### 5.3  Ground-truth labels — what's actually knowable

Not all role labels have ground truth.  What we *can* derive:

- `agent` identity — via `ControlledActorClaim` (colour + background
  signature) once the COS loop has committed one on this game.  This
  is the agent identity the engine itself believes after behavioural
  evidence, independent of any LLM claim.
- `target` position — via the cell the winning episode's final step
  landed on, from the adapter's frame history.
- `wall` classification — via `ActorTransitionClaim(pre, action,
  delta=0)` at the adjacent cells.

Other roles (hazard, resource, decor) have no symbolic ground truth
and are scored tutor-only.  Spec this up front so eval results
aren't misread.

### 5.4  Held-out games — Sonnet smoke-pass first

Before running DD eval, confirm each held-out `game_id` passes a
Sonnet smoke test (tutor can at least enumerate and propose
linkage).  If Sonnet itself fails on a game, DD has no tutor answer
to match and scoring is meaningless.  Smoke pass is integrated into
the tutor-probe script (§1.3): if the probe returns zero cache
entries for a game, that game is marked `tutor_broken` and excluded
from the held-out eval set.

**Eval script:** `arc_agi_3/tools/eval_dd.py` — reads three cache
dirs (tutor + PUPIL-raw + PUPIL-DD), parses replies, computes Tier-1
and Tier-2 metrics, prints a table with PUPIL-DD lift over PUPIL-raw
and gap-to-tutor.

---

## Part 6 — What the COS development session must build (checklist)

These are the items the COS session needs to land, roughly in order,
to enable a DD eval.  Load-bearing items (§review) are marked ★.

### Now — unlocks everything downstream
- [x] `CachedChatBackend` with per-call JSON files — done (`1ea1d56`)
- [x] `PROPOSE_GOAL_LINKAGE` trigger (GAP 7) — done (engine `fe8feac`
      / arc-agi-3 `d83967e`)
- [ ] ★ **Metadata-at-write-time on `CachedChatBackend`** (§1.2)
      – add `metadata` attribute; merge into stored JSON on miss
      – small change in `arc_agi_3/backends/cached.py`
- [ ] ★ **Loop runner populates `backend.metadata`** before each
      `run_episode` with all fields listed in §1.2
- [ ] ★ **Prompt-template SHA exposure** — each trigger / question
      builder publishes `PROMPT_SHA` (SHA-1 of the template string)
      so metadata can record it.  Lives in
      `cognitive_os/oracle.py` (for engine triggers) and
      `arc_agi_3/{observer,mediator}.py` (for question builders).
- [ ] ★ **Episode-outcome back-patcher** — after each `run_episode`
      returns, walk cache entries whose `metadata.episode_id ==
      <just-finished>` and set `metadata.episode_outcome` to
      `win`/`loss`/`timeout`/`crashed`.  One idempotent helper in the
      loop runner.

### Now — unblocks corpus growth
- [ ] ★ **Tutor-probe script** `.tmp/tutor_probe.py` (§1.3) — fires
      initial-frame triggers across N games with `max_steps=2`,
      produces 2 cache entries per game in ~5s.  This is the
      primary corpus-growth mechanism until L1 is solved on
      several games.
- [ ] Multi-game loop runner — parameterise `solve_l1_loop.py` for
      use after L1 is solved.  Lower priority than tutor-probe.

### After L1 is solved on ≥1 game
- [ ] Run loop runner on ≥5 games (mix of probe-only and full-episode
      entries).  Corpus target: ≥15 `ENUMERATE_OBJECTS`, ≥10
      `PROPOSE_GOAL_LINKAGE`.

### Before DD eval
- [ ] `arc_agi_3/tools/build_kf.py` — §3; with rank-normalised
      palette, validation filter, dedup, skipped-entry accounting
- [ ] `arc_agi_3/backends/ollama_backend.py` (or `vllm_backend.py`) —
      OSS PUPIL backend implementing `chat()` (§4.2)
- [ ] `arc_agi_3/backends/dd_backend.py` — KF retrieval + injection
      wrapper with per-game cap, validated-only default,
      min_similarity floor (§4)
- [ ] `arc_agi_3/tools/eval_dd.py` — three-arm comparison
      (Tutor / PUPIL-raw / PUPIL-DD), Tier-1 and Tier-2 metrics (§5)

---

## Scope boundaries (do not violate)

- **No fine-tuning.**  PUPIL is the stock OSS model; the KF is a
  retrieval layer in the prompt only.
- **Minimal engine-adjacent changes, clearly bounded.**  Prior spec
  revision claimed "no engine changes" — this was incorrect.  DD
  requires (a) metadata-at-write-time on `CachedChatBackend`
  (arc-agi-3-side, not engine-side), (b) a `PROMPT_SHA` constant per
  trigger and question-builder module (engine-side for oracle
  triggers, adapter-side for question builders).  Both are
  additive; no existing behaviour changes.  `oracle.py`,
  `observer.py`, `mediator.py`, `episode_runner.py` gain metadata
  publishing surfaces but no logic changes.
- **No new question types for DD itself.**  If PUPIL is weak on an
  existing question type, the fix is a richer KF entry, not a new
  enum value.  (New question types may still land for unrelated
  capability work.)
- **Observer and Mediator KF pools are separate.**  Question types
  are the natural shard key; mixing them dilutes retrieval.
- **Colour literals are forbidden in heuristic text.**  Rank-only
  (§2.1).  Build-script guard rejects any heuristic containing a
  literal colour ID numeric.
- **Tutor is not oracle.**  Every eval gate is rooted in
  ground-truth outcomes (§5.2 Tier 1) where available; agreement
  with tutor is a diagnostic, not a target.
- **Backend swap is the primary visible change to the adapter.**
  The adapter instantiates a `DDChatBackend(OllamaBackend(...))` in
  place of `CachedChatBackend(AnthropicBackend(...))`.  The new
  metadata attribute must also be populated by whichever harness is
  driving `run_episode`.
