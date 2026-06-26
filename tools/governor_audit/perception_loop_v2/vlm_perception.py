"""VLM-driven perception orchestration loop.

For each turn:

  1. Run the deterministic detector + temporal registry to produce a
     Scene with up-to-date tracks, behaviour events, and spatial
     relationships.
  2. Serialise the Scene into a compact JSON summary the VLM reads.
  3. Build a system + user prompt that gives the VLM the operational
     context, the closed tool vocabulary, and a turn-specific
     question ("classify the active entities in this frame, given
     the recent history").
  4. Hand the frame image + prompt to the VLM backend.
  5. Parse the VLM's JSON response.
  6. Apply the annotations back to the Scene via annotate_entity.

The first cut is **single-shot per turn** — the VLM emits one JSON
response with all its annotations, no follow-up tool calls.  Adding
a multi-turn tool-call loop (where the VLM asks for entity_history,
spatial_relationships, etc. before committing) is a future
refinement; the tool surface (vlm_tools.py) is ready when we want
to wire it in.

For now this produces structured entity classifications the planner
and validator can read from the Scene.
"""

from __future__ import annotations

import base64
import json
import re
import sys
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .temporal_registry import Scene
from . import vlm_tools as T
from .vlm_backend import VLMBackend
from .hypothesis_bridge import propose_track_property


# ---------------------------------------------------------------------------
# Prompts -- LIGHT operational context, NO game-specific mechanics.  Same
# discipline as the vlm_experiment harness (feedback_vlm_operational_context).
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """\
You are the perception module of an agent playing an interactive
grid-based puzzle game.  Each turn you receive (1) the current
frame, (2) a structured summary of what the perception system
already knows about the scene from prior turns, and (3) a question.

Operational context (game-agnostic, true for every game in this
family):
  - The scene contains a small avatar called the AGENT.  It moves
    around the play area between turns in response to ACTIONS.
  - The scene contains one or more BACKGROUND regions (floor, wall,
    interior, exterior) that fill most of the frame.
  - The scene may contain other distinct visual elements: things
    the agent can collect, things that threaten the agent, on-screen
    status indicators (counters, meters), scenery, decoration.
  - The game runs turn-by-turn.  Across many turns, persistent UI
    elements stay in roughly the same place; transient sprites
    appear and disappear; collectables vanish when the agent
    contacts them.

Your job is to classify each ACTIVE TRACK the perception system
has identified.  You will receive:
  - A list of tracks (each with a track_id, current position,
    pixel count, signature key, behaviour history, neighbours).
  - The frame image showing the scene at this turn.

Your output is a JSON array of entity-classification objects, one
per track.  Each object has these fields (use exactly these names):

  track_id          (int)    - the track you're classifying
  description       (string) - a one-line description of what
                                you see at this track's location
  role_label        (string) - one of: "agent" | "collectable" |
                                "threat" | "status_indicator" |
                                "win_marker" | "scenery" | "unknown"
  role_confidence   (string) - "low" | "medium" | "high"
  evidence          (string) - one short sentence justifying the
                                role label, citing observed
                                behaviour or visual features
  properties        (object) - typed key-value facts you can read
                                off the visuals/history: keys you
                                may use include "color",
                                "shape", "size_class", "movement"

Output ONLY the JSON array.  No prose, no markdown fences.  If a
track's identity is genuinely ambiguous, set role_label to "unknown"
and confidence to "low" -- do not guess.
"""


USER_PROMPT_TEMPLATE = """\
Turn {turn} of the game.

Scene summary (perception's view of what's been observed so far,
truncated to the most relevant context):

{scene_json}

Available tool name vocabulary (for reference; this is what your
output's role_label etc. choices map onto): {tool_names}

Question: For each active track in the scene summary, return one
classification object as specified in the system prompt.  Skip
tracks marked present_now=false.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rgb_to_b64(rgb: np.ndarray, *, upscale: int = 4) -> str:
    """Encode an RGB ndarray as a base64-encoded PNG, optionally
    upscaled with nearest-neighbour so small sprites are legible.
    """
    img = Image.fromarray(rgb)
    if upscale > 1:
        img = img.resize(
            (img.size[0] * upscale, img.size[1] * upscale),
            Image.NEAREST,
        )
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _strip_code_fence(text: str) -> str:
    """Strip ```json ... ``` fences if the VLM wrapped its JSON."""
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines)
    return s


def parse_vlm_response(text: str) -> list[dict]:
    """Parse the VLM's JSON-array response into a list of dicts.
    Returns [] if parsing fails (with the raw response available for
    debugging via the caller).
    """
    s = _strip_code_fence(text)
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        # Last-resort: find the first [...] in the text.
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
    return []


# ---------------------------------------------------------------------------
# Orchestration loop
# ---------------------------------------------------------------------------


@dataclass
class TurnResult:
    """One turn's VLM perception output."""
    turn: int
    raw_response: str
    parsed_annotations: list[dict] = field(default_factory=list)
    applied_track_ids: list[int] = field(default_factory=list)
    parse_error: Optional[str] = None


@dataclass
class PerceptionVLMLoop:
    """Drives the per-turn VLM perception calls and applies the
    returned annotations to the Scene.

    When `world_state` is provided, VLM-proposed role_labels also
    propose PropertyClaim(track, "role", role_label) into the
    hypothesis_store and store the hypothesis_id on the track.
    Without a WorldState, the loop still works (annotations land
    on the track) but no hypotheses are minted.
    """

    scene: Scene
    backend: VLMBackend
    # Where to save per-turn results (for tracing / debugging).
    output_dir: Optional[Path] = None
    # History window passed to serialize_for_vlm.
    history_window: int = 8
    # Image upscale factor for the frame sent to the VLM.
    image_upscale: int = 4
    # Optional WorldState for hypothesis-store wiring.  When None,
    # role labels are stored as category_labels only; no PropertyClaims
    # are minted.
    world_state: Optional[object] = None
    # Source tag for the hypothesis_store's source field.
    vlm_source: str = "vlm:perception"

    def __post_init__(self) -> None:
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def perceive_turn(
        self,
        turn: int,
        rgb_frame: np.ndarray,
    ) -> TurnResult:
        """Build the prompt, call the VLM, parse the response, apply
        annotations to the Scene.  Caller must have already ingested
        this turn's entities into the Scene before invoking.
        """
        # Build the scene summary the VLM will read.
        summary = self.scene.serialize_for_vlm(
            turn, history_window=self.history_window,
        )
        # Trim the summary to active entities + recent context so the
        # prompt isn't huge.  serialize_for_vlm already does most of
        # the curation; we just drop entity fields that the VLM
        # doesn't need to see (bbox_logical pixel-precision).
        for e in summary.get("entities", []):
            e.pop("current_bbox_logical", None)
        scene_json = json.dumps(summary, indent=2, default=str)

        user_prompt = USER_PROMPT_TEMPLATE.format(
            turn=turn,
            scene_json=scene_json,
            tool_names=", ".join(T.TOOL_NAMES),
        )

        image_b64 = _rgb_to_b64(rgb_frame, upscale=self.image_upscale)
        raw_text = self.backend.call(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            image_b64=image_b64,
        )

        annotations = parse_vlm_response(raw_text)
        parse_error = None
        if not annotations and raw_text.strip():
            parse_error = "response did not parse as a JSON array"

        applied: list[int] = []
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            tid = ann.get("track_id")
            if not isinstance(tid, int):
                continue
            description = ann.get("description")
            role_label = ann.get("role_label")
            role_confidence = ann.get("role_confidence", "medium")
            props = ann.get("properties")
            if not isinstance(props, dict):
                props = None
            # Apply via the canonical write tool.
            label_pair: Optional[tuple[str, str]] = None
            if isinstance(role_label, str):
                label_pair = (role_label, str(role_confidence))
            T.annotate_entity(
                self.scene, tid,
                description=description if isinstance(description, str) else None,
                add_category_label=label_pair,
                set_properties=props,
            )
            # If a WorldState is wired in, also propose a PropertyClaim
            # so the role hypothesis lives in the hypothesis_store with
            # proper credence tracking.  The returned hypothesis_id is
            # stored on the track by propose_track_property itself.
            if self.world_state is not None and isinstance(role_label, str):
                rationale = ann.get("evidence") if isinstance(ann.get("evidence"), str) else None
                propose_track_property(
                    self.world_state, self.scene, tid,
                    property="role", value=role_label,
                    source=self.vlm_source, step=turn,
                    rationale=rationale,
                )
            applied.append(tid)

        result = TurnResult(
            turn=turn,
            raw_response=raw_text,
            parsed_annotations=annotations,
            applied_track_ids=applied,
            parse_error=parse_error,
        )

        # Persist artifacts for tracing.
        if self.output_dir is not None:
            (self.output_dir / f"turn_{turn:03d}_raw.txt").write_text(
                raw_text, encoding="utf-8",
            )
            (self.output_dir / f"turn_{turn:03d}_parsed.json").write_text(
                json.dumps(annotations, indent=2), encoding="utf-8",
            )
            (self.output_dir / f"turn_{turn:03d}_scene_summary.json"
                ).write_text(scene_json, encoding="utf-8")

        return result
