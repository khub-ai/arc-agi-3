# Operational primitives catalog

The catalog is the set of game-agnostic primitives that the perception
VLM uses to identify entity roles and infer mechanics on any game
in the ARC-AGI-3 family.  One file per primitive, organised by
subdirectory according to its `primitive_kind`.

## Directory layout

```
catalog/
  interaction/         # interaction primitives the player can do
  entity_role/         # roles entities can play in a game
  match_condition/     # how the game decides a win is achieved
  relationship_kind/   # kinds of inter-entity relationships
  state_change_effect/ # effects that triggers / actions produce
```

## Per-primitive schema

Each `.json` file is a single primitive entry:

- `primitive_id` — closed-vocabulary string; matches the filename
  (without `.json`).  Used by the planner and as a tag in
  perception output.
- `primitive_kind` — one of `interaction`, `entity_role`,
  `match_condition`, `relationship_kind`, `state_change_effect`.
  Must match the subdirectory the file lives in.
- `description` — operator-facing prose explaining what the
  primitive is.  Documentation, not part of the VLM prompt.
- `vlm_recognition_hints` — short list of 1-3 sentences the VLM
  gets in its prompt to recognise instances of this primitive in
  a frame.  Static visual cues + typical behavioural cues.  Keep
  compact; this is what bloats the prompt if let loose.
- `interaction_signature` (optional) — typed fields describing
  trigger, effect, observable change.  Used to confirm a
  perceived primitive via observation.
- `planner_consumption` (optional) — fields the planner reads to
  know what to do with a confirmed instance: plan-tree node kind,
  directive template, compatible match conditions.
- `anchored_in_samples` — names of `tests/perception_samples/<sample>/`
  directories where this primitive was operator-validated.
  Audit trail for extension safety.
- `extension_notes` (optional) — open questions about edge cases,
  related primitives, generalisations.

## Adding a new primitive

1. Pick the right subdirectory.
2. Create `<primitive_id>.json` with all required fields.
3. Add `anchored_in_samples` entries for every sample where the
   operator has confirmed an instance.
4. Run the perception test runner; verify the new primitive
   doesn't regress recognition on existing samples.

## Why per-file rather than one big catalog

Each primitive can be reviewed and extended in isolation.  Diffs
stay small.  Adding a primitive is one new file rather than a
patch to a shared list.  The runtime loader walks the directory
and composes the prompt from all entries; no manual registration.
