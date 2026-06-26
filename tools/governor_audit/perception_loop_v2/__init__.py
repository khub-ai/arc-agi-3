"""Governor-Perception loop, v2.

Replaces the retracted v1 palette-keyed implementation.  All rule
bodies and detector inputs use substrate-agnostic primitives only:
visual signature (pixel-derived descriptors), persistence / motion,
interaction response, spatial pattern.  See P10 in
docs/DURABLE_PRINCIPLES.md and the retraction notes in
docs/SPEC_governor.md.

This package contains:

  observation.py  — substrate-agnostic per-turn Observation records.
  classifier.py   — applies learned rules to produce per-cell codes.
  detector.py     — proposes rules from observation patterns.
  aggregator.py   — commits rules through sandbox/trial/established.
  apply.py        — composes detector + aggregator into a runnable loop.
  score.py        — per-cell agreement against truth.json.
  run.py          — end-to-end driver for fixture data.

Discipline (see DEVELOPMENT_DISCIPLINE.md):

  * truth.json is the answer key for scoring; the loop NEVER reads it.
  * The substrate-agnostic interface is enforced at the Observation
    boundary — no module under this package may import palette-keyed
    helpers (decode_palette_indices, ARC_PALETTE) for the purpose
    of building rules.  RGB pixels and derived visual descriptors
    are the only acceptable substrate-side features.
  * Every schema feature in a learned rule body must be motivated
    by what the detector can autonomously propose.
"""
