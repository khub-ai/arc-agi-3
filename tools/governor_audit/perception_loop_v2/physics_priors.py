"""Physics-world priors — a small open-vocabulary library of
real-world-physics hypotheses that the strategy actor consults at
trial start (and whenever a new entity type is detected).

DISCIPLINE: many games in this family simulate physical-world
mechanics.  When perception detects entities with shapes that
match a real-world object (spear / arm / block / button /
container / rope / gear), the actor should consider what that
object DOES in the real world as a starting hypothesis BEFORE
treating the game as an abstract puzzle.

Substrate role: hold a small library of free-form prior
descriptions, each tied to an open-vocabulary shape / role
signature.  Surface matching priors in the strategy prompt.
The actor decides whether the prior fits and whether to commit
a hypothesis based on it.  No game-specific knowledge — each
prior names a class of REAL-WORLD OBJECT and the behaviors that
class exhibits, not a specific game's mechanic.

Each prior is intentionally TERSE.  The actor reads it as a
seed for hypothesis generation, not as a hard rule.

Persistence: priors live on a JSON file alongside subroutine_kb
so they can grow across runs.  Defaults shipped here cover the
common shapes encountered in pixel-art puzzle games.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

DEFAULT_PHYSICS_PRIORS_PATH = Path(
    __file__
).parent.parent.parent / ".tmp" / "physics_priors.json"


@dataclass
class PhysicsPrior:
    """A single open-vocabulary physics prior.

    Each prior describes a real-world OBJECT CLASS and the
    behaviors it exhibits.  The actor matches it against
    perceived entities and considers whether to commit a
    MechanicHypothesis / WinConditionHypothesis seeded by it.
    """
    prior_id: str
    name: str
        # short label, e.g. "spear / skewer / piercing rod"
    shape_signature: str
        # free-form: "long thin structure attached to agent,
        #  one tip away from agent's body"
    behaviors: str
        # free-form description of what objects of this class
        # do in the physical world.  Multiple behaviors comma-
        # or newline-separated.
    matches_against: str
        # free-form: which perceived entity roles / appearances
        # this prior plausibly applies to.  e.g. "horizontal_arm,
        #  rod, beam, decoration with thin elongated bbox"
    note: str = ""
        # optional aphorism / framing


def _default_priors() -> list[PhysicsPrior]:
    """Seed library — small and game-agnostic.  Each prior names
    a real-world object class.  The actor decides applicability."""
    return [
        PhysicsPrior(
            prior_id="prior_spear_skewer",
            name="spear / skewer / piercing rod",
            shape_signature=(
                "long thin rigid structure with one end fixed to "
                "an agent / wielder and the other end (the TIP) "
                "moving freely in space"
            ),
            behaviors=(
                "EXTEND: tip can be advanced into open space.  "
                "PUSH: when the tip contacts an UNSUPPORTED object, "
                "the object slides away in the tip's motion direction.  "
                "PIERCE: when the tip contacts an object that is "
                "SUPPORTED FROM THE OPPOSITE SIDE (by a wall, "
                "another solid object, or the ground), the object "
                "becomes IMPALED on the shaft and travels with the "
                "spear thereafter.  PUSH AND PIERCE ARE A SINGLE "
                "BINARY GATED BY SUPPORT.  STACK: multiple "
                "impaled objects accumulate on the shaft, ordered "
                "from base (near wielder) to tip in the sequence "
                "they were pierced.  RELEASE: typically irreversible "
                "without an explicit unstack mechanic."
            ),
            matches_against=(
                "any thin elongated structure attached to the agent; "
                "common appearances include arm, rod, beam, harpoon, "
                "lance, telescoping pole, fishing rod"
            ),
            note=(
                "If a perceived entity matches this signature, the "
                "actor's first hypothesis about how it interacts "
                "with neighbour blocks SHOULD include the "
                "push-vs-pierce binary gated by neighbour support — "
                "this is the single most common physical mechanic "
                "in pixel-art puzzle games featuring a manipulator."
            ),
        ),
        PhysicsPrior(
            prior_id="prior_pushable_block",
            name="pushable block",
            shape_signature=(
                "compact rigid shape, roughly square or "
                "rectangular, distinct from background, NOT "
                "attached to the agent"
            ),
            behaviors=(
                "PUSHABLE: when contacted by a moving solid (agent, "
                "tip of an arm, another block), slides in the "
                "direction of contact UNLESS something solid backs "
                "it up.  CHAIN-PUSH: when several pushable blocks "
                "are colinear, pushing the first chain-pushes the "
                "rest until the last one hits a wall or void.  "
                "STACKABLE: vertically aligned blocks stack on top "
                "of each other and lift together when the bottom "
                "one is raised.  GRAVITY: may fall under gravity if "
                "vertically unsupported (game-dependent)."
            ),
            matches_against=(
                "small coloured blocks, tiles, crates, gems, balls, "
                "collectables that aren't immediately consumed on "
                "contact"
            ),
            note=(
                "Push-vs-pierce binary lives under the spear prior; "
                "the block prior is the PASSIVE side of that "
                "interaction.  When both priors fire on the same "
                "scene, the actor should combine them: the spear "
                "pushes UNSUPPORTED blocks and pierces SUPPORTED "
                "ones."
            ),
        ),
        PhysicsPrior(
            prior_id="prior_wall_boundary",
            name="wall / boundary / immovable barrier",
            shape_signature=(
                "extended solid region along the playfield edge "
                "OR a straight thick line bisecting the playfield"
            ),
            behaviors=(
                "BLOCKS MOTION: solid objects stop when they reach "
                "the wall.  BRACES: a block pressed against a wall "
                "from outside has the wall as a support, so a "
                "spear tip contacting the block from the other side "
                "PIERCES rather than pushes.  BOUNDS THE FIELD: the "
                "playfield is everything inside the wall."
            ),
            matches_against=(
                "playfield border, divider rows, fences, fixed thick "
                "structures that don't move"
            ),
            note=(
                "The wall is the second-most-important spear "
                "interaction partner after blocks: it provides "
                "support that turns push into pierce."
            ),
        ),
        PhysicsPrior(
            prior_id="prior_status_indicator",
            name="status indicator / HUD readout",
            shape_signature=(
                "small fixed-position icons or readouts at the edge "
                "of the playfield, often colour-coded, in a strip"
            ),
            behaviors=(
                "READS STATE: the indicator's visual state (colour, "
                "fill, outline) reflects some hidden game state, "
                "updating in response to game events.  TRACKS "
                "PROGRESS: a count, a set of toggles, or a sequence "
                "of slots; a full indicator typically corresponds "
                "to a goal condition.  REACTIVE: the indicator "
                "changes WHEN the relevant event happens, so it is "
                "the cheapest ground-truth signal for verifying a "
                "hypothesis about what triggers progress."
            ),
            matches_against=(
                "icons in the playfield border, fixed-position dots "
                "or bars, colour-keyed tiles arranged in a strip"
            ),
            note=(
                "CRITICAL: HUD changes are a ground-truth signal "
                "for hypothesis validation.  The perception layer "
                "should be configured to pixel-diff inside HUD "
                "bboxes per turn so HUD visual events become "
                "first-class observations.  Without this, every "
                "'achieved' claim is inference, not observation."
            ),
        ),
        PhysicsPrior(
            prior_id="prior_gravity",
            name="gravity / falling",
            shape_signature=(
                "tall playfield with no obvious horizontal support "
                "lines, and objects observed moving downward "
                "without input"
            ),
            behaviors=(
                "OBJECTS FALL: unsupported objects move toward the "
                "bottom of the playfield each turn.  SUPPORT: "
                "objects on top of others, or on the floor / wall, "
                "stay put.  CHAIN-FALL: removing support causes "
                "everything above to fall."
            ),
            matches_against=(
                "tall vertical scenes with stacked objects, "
                "platformers, falling-puzzle layouts"
            ),
            note=(
                "Even without observation, the actor should probe "
                "this once early by stepping aside and watching."
            ),
        ),
    ]


def load_priors(
    path: Path = DEFAULT_PHYSICS_PRIORS_PATH,
) -> list[PhysicsPrior]:
    """Load the physics-priors library.  Auto-seeds defaults if
    the file doesn't yet exist."""
    if not path.exists():
        priors = _default_priors()
        save_priors(priors, path)
        return priors
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _default_priors()
    return [PhysicsPrior(**rec) for rec in data.get("priors", [])]


def save_priors(
    priors: list[PhysicsPrior],
    path: Path = DEFAULT_PHYSICS_PRIORS_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"priors": [asdict(p) for p in priors],
            "saved_at": time.time()}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    tmp.replace(path)


def format_physics_priors_surface(
    priors: Optional[list[PhysicsPrior]] = None,
    *,
    max_priors: int = 5,
) -> str:
    """Render the priors library as a strategy-prompt block.

    Game-agnostic — each prior names a real-world object class
    and its behaviors.  The actor reads the list and decides
    whether any apply to perceived entities, treating matches as
    seeds for MechanicHypothesis / WinConditionHypothesis."""
    if priors is None:
        priors = load_priors()
    if not priors:
        return ("  (no physics priors registered.  See "
                "physics_priors.py to seed defaults.)")
    lines: list[str] = []
    lines.append(
        f"  {len(priors)} PHYSICS-WORLD PRIOR(s) registered.  "
        "Each names a real-world object class and its behaviors.  "
        "READ THEM as candidate hypotheses for HOW entities in "
        "this game might behave.  When a perceived entity matches "
        "a prior's shape, your DEFAULT hypothesis about that "
        "entity's mechanic should be the prior's described "
        "behaviors UNTIL observation falsifies it.  Many games "
        "in this family simulate real-world physics; using the "
        "prior costs nothing and skips dozens of wrong-direction "
        "probes."
    )
    for p in priors[:max_priors]:
        lines.append("")
        lines.append(
            f"  PRIOR id={p.prior_id!r}  ({p.name})"
        )
        lines.append(f"    Shape signature:  {p.shape_signature}")
        lines.append(f"    Behaviors:        {p.behaviors}")
        lines.append(f"    Matches against:  {p.matches_against}")
        if p.note:
            lines.append(f"    Note:             {p.note}")
    lines.append("")
    lines.append(
        "  HOW TO USE: when your perception snapshot includes an "
        "entity matching a prior's shape, consider committing a "
        "MechanicHypothesis seeded by the prior's behaviors.  "
        "Then probe SPECIFICALLY to confirm or falsify it.  Do "
        "NOT treat the prior as game-specific knowledge — it is "
        "a hypothesis to be validated."
    )
    return "\n".join(lines)
