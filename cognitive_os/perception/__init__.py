"""VLM-driven perception layer.

Two-layer design (see docs/SPEC_perception.md when written):

* Layer A — deterministic pixel geometry (pixel_elements + entity
  fingerprints).  Game-agnostic, runs every turn.
* Layer B — VLM-driven semantic labelling.  Consults the operational
  primitives catalog (this module's ``catalog/``) plus the current
  frame and produces typed entity records with roles, relationships,
  and match conditions.

Module surface (filled in as we implement):

* ``catalog`` package — operational primitives the VLM uses to
  classify entities and infer mechanics.  Organised by
  primitive_kind subdirectories.
* ``vlm_client`` — wraps the VLM call.
* ``prompt`` — assembles the Layer B prompt from catalog + frame +
  Layer A geometry.
* ``parser`` — validates and normalises the VLM's JSON response.
* ``test_runner`` — runs the pipeline on
  ``tests/perception_samples/<sample>/`` and compares against the
  operator-authored ``ground_truth.json``.
"""
