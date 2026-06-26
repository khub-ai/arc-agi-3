"""Game- and domain-agnostic DEBUGGING DISCIPLINE.

When an OBSERVATION contradicts an EXPECTATION (a stated prediction, or a belief
the actor is relying on), that mismatch is a SURPRISE -- a signal that one of the
ASSUMPTIONS the belief rests on is WRONG.  The competent response is NOT to
re-guess a new story.  The recurring failure this prevents: concluding
"no effect" / "canned" / "stuck" / "not the mover" / "impossible" from a SUMMARY
or a single SETTLED frame, without taking the situation apart and inspecting the
raw detail (the animation frames, the exact pixels, the panel state).

The fix is the universal DEBUGGING skill: when something does not add up,
DECOMPOSE it into its component assumptions and VERIFY each one with a minimal
controlled check, isolating the broken piece -- the same discipline that debugs a
program (bisect to the failing step), a car (check each subsystem in turn), or a
puzzle (verify each mechanic in isolation).  This module (a) DETECTS the surprise
from the expectation-vs-observation polarity, and (b) surfaces the
decompose-and-verify PROTOCOL so the actor DEBUGS rather than re-concludes.

Pure text/heuristics; no model cost.  Game-agnostic: it keys on generic outcome
polarity (did SOMETHING change vs nothing), never on any game's specifics.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

# Words that mean the actor EXPECTED an effect, vs expected nothing/no-op.
_EFFECT_WORDS = (
    "move", "moved", "moves", "shift", "travel", "climb", "descend", "rise",
    "fall", "slide", "change", "changes", "changed", "increase", "increases",
    "advance", "appear", "appears", "disappear", "vanish", "grow", "shrink",
    "rotate", "fill", "unite", "mate", "merge", "win", "wins", "solve",
    "score", "light", "lights", "toggle", "set", "open", "close", "cross",
    "reach", "reaches", "commit", "execute", "executes", "trigger",
)
_NO_EFFECT_WORDS = (
    "no-op", "no op", "nothing", "unchanged", "stays", "stay", "no change",
    "no effect", "inert", "blocked", "canned", "reverts", "revert", "no move",
    "does not move", "doesn't move", "will not", "won't", "remains",
)
_NEGATORS = ("not ", "n't", "no ", "never ", "without ")


def _polarity_from_text(text: str) -> Optional[str]:
    """'effect' | 'no_effect' | None from a free-text prediction/summary."""
    if not text:
        return None
    t = " " + text.lower() + " "
    no = any(w in t for w in _NO_EFFECT_WORDS)
    eff = any(re.search(r"\b" + re.escape(w) + r"\b", t) for w in _EFFECT_WORDS)
    if no and not eff:
        return "no_effect"
    if eff and not no:
        return "effect"
    return None                                  # mixed / unclear -> don't claim a polarity


def expectation_polarity(prediction_text: Optional[str]) -> Optional[str]:
    return _polarity_from_text(prediction_text or "")


def observation_polarity(*, summary: Optional[str] = None,
                         visual_events=None, agent_moved=None,
                         score_increased=None, entities_changed=None,
                         settled_change_cells: Optional[int] = None) -> Optional[str]:
    """Did SOMETHING happen?  Prefers HARD signals (a measured change) over the
    text summary.  Returns 'effect' | 'no_effect' | None."""
    hard_effect = bool(visual_events) or bool(agent_moved) or bool(score_increased) \
        or bool(entities_changed) or bool(settled_change_cells)
    if hard_effect:
        return "effect"
    # No hard signal: fall back to the summary's polarity, but a hard 'no signal'
    # plus a summary that ALSO reads as no-effect is a confident no_effect.
    sp = _polarity_from_text(summary or "")
    if sp == "no_effect":
        return "no_effect"
    # hard signals absent but summary claims an effect -> UNRESOLVED (the summary
    # may be asserting an effect the measured signals didn't capture -- exactly
    # the case to debug, not to trust either way).
    return None if sp == "effect" else "no_effect"


def detect_surprise(prediction_text: Optional[str], *, summary=None,
                    visual_events=None, agent_moved=None, score_increased=None,
                    entities_changed=None, settled_change_cells=None) -> Optional[dict]:
    """A SURPRISE = the expectation's polarity and the observation's polarity
    disagree (e.g. predicted an effect, measured nothing -- or predicted a no-op
    and something changed).  Returns an anomaly dict or None.  Conservative: only
    fires when BOTH polarities are confidently known and they conflict."""
    exp = expectation_polarity(prediction_text)
    obs = observation_polarity(summary=summary, visual_events=visual_events,
                               agent_moved=agent_moved, score_increased=score_increased,
                               entities_changed=entities_changed,
                               settled_change_cells=settled_change_cells)
    if exp and obs and exp != obs:
        return {"expected": exp, "observed": obs, "prediction": prediction_text}
    return None


# The standing reasoning principle (for the strategy SYSTEM prompt).
DISCIPLINE = (
    "DEBUG-DON'T-GUESS (a universal skill, not game-specific) — when an "
    "observation does NOT match what you expected, or two of your beliefs "
    "conflict, that mismatch means ONE OF YOUR ASSUMPTIONS IS WRONG. "
    "DEFAULT ATTRIBUTION — assume the bug is on YOUR side. The environment is "
    "almost always consistent and beatable (these games are designed to be easy "
    "for a human); when something 'doesn't work', the weight of probability is "
    "that YOUR pipeline is wrong — a mis-tracked entity, a biased centroid, a "
    "flipped row/col or pixel/tick convention, an aim computed from the wrong "
    "position, a mis-read panel — NOT that the game is impossible or has a "
    "hidden mechanic you can't reach. Suspect and check your OWN perception / "
    "control / measurement FIRST, exhaustively, before concluding the level "
    "can't be done or blaming an invisible trigger. Do NOT "
    "react by inventing a new story or by concluding 'no effect / canned / "
    "stuck / not the mover / impossible' from a SUMMARY or a single settled "
    "frame. DEBUG it: (1) STATE the inference that just failed and DECOMPOSE it "
    "into the chain of independent assumptions it rests on (e.g. 'the action "
    "executed', 'I set the control correctly', 'the thing I tracked is the one "
    "that moves', 'I read the ANIMATION, not just the settled frame', 'I "
    "measured the pixels, not a label'). (2) For EACH assumption design the "
    "MINIMAL controlled check that confirms or refutes it — vary ONE thing and "
    "inspect the RAW evidence (the per-frame animation, the exact pixels, the "
    "panel cell states), never a summary. (3) Run the checks to LOCALIZE which "
    "assumption is false. (4) Re-conclude ONLY from the verified pieces. This is "
    "the same discipline that debugs a program (bisect to the failing step), a "
    "car (test each subsystem), or any puzzle — take it apart and check each "
    "piece works as expected before drawing a conclusion. "
    "EXHAUST THE VISIBLE CLUES BEFORE YOU SURRENDER. NEVER conclude 'no action "
    "works' / 'stuck' / 'unwinnable' while a readily-visible on-screen element "
    "remains un-probed — the game is DESIGNED to be solvable, so an element you "
    "can see but haven't actively tested is the single most likely missing key. "
    "Two rules make this real: (1) MEASURE, don't eyeball — decide whether an "
    "action had an effect from the PIXELS (frame-diff / locate_entity), never "
    "from a glance at the rendered frame (you WILL misread a subtle 1-step move "
    "or a margin-offset position). (2) VARY THE PARAMETER, especially NEAR each "
    "element — a click's effect usually depends on WHERE it lands: ON an element, "
    "ADJACENT to it (a few ticks away), and BETWEEN two elements are different "
    "probes. A handful of centre/far clicks showing 'nothing' is NOT evidence "
    "the control is inert — effects like click-to-move, aim, select, or merge "
    "trigger only when you act CLOSE to a thing. Systematically test interacting "
    "with and AROUND every salient element, measure each, and only then escalate."
)


# =============================================================================
# TREND-AWARE self-monitoring (the drift/stall a single-step check cannot see).
#
# detect_surprise above is SINGLE-STEP and POLARITY-only: it asks "did SOMETHING
# change vs nothing?".  It is blind to a slow DIRECTIONAL drift -- every step has
# the right polarity (the mover DID move) yet the series veers off target, or the
# distance-to-goal quietly stops shrinking.  Self-correction must watch the TREND,
# not each step.  This monitor keeps per-metric time series the actor expects to
# trend a certain way and fires when the TREND is violated -- so COS debugs on its
# own, without being told the run went wrong.
#
# Detection is STRUCTURAL (sign / monotonicity over a short window) -- no tuned
# magnitude thresholds (a banned magic number); `window` is just "how many in a
# row to be sure", not a calibrated quantity.
# =============================================================================


@dataclass
class ProgressLedger:
    """Per-metric time series of (turn, value) the actor expects to TREND a
    certain way.  Domain-agnostic: the driver feeds whatever scalars it can
    measure each turn (distance-to-goal, deviation-from-expected-path, ...)."""
    series: dict = field(default_factory=dict)

    def record(self, name: str, turn: int, value: float) -> None:
        s = self.series.setdefault(name, [])
        if s and s[-1][0] == turn:               # one sample per turn (idempotent)
            s[-1] = (turn, float(value))
        else:
            s.append((turn, float(value)))


def detect_trend_failure(series, expect: str, *, window: int = 3) -> Optional[dict]:
    """STRUCTURAL drift/stall over the last `window` steps of a (turn,value)
    series.  No tuned magnitude threshold -- pure sign / monotonicity.

      expect='decrease' (e.g. distance-to-goal): FAIL 'not_converging' when the
        value did NOT decrease on ANY of the last `window` steps -- the actor
        keeps acting but is not getting closer (a stall, or an outright
        regression).
      expect='zero' (e.g. a deviation that should stay ~0): FAIL 'drift' when
        |value| GREW on every one of the last `window` steps with a CONSTANT sign
        -- a systematic, accumulating departure from the expected path.

    Returns a verdict dict or None."""
    vals = [float(v) for _, v in series][-(window + 1):]
    if len(vals) < window + 1:
        return None
    d = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    if expect == "decrease":
        if all(x >= 0 for x in d):               # never improved across the window
            kind = "regression" if all(x > 0 for x in d) else "stall"
            return {"failure": "not_converging", "kind": kind,
                    "window": window, "from": vals[0], "to": vals[-1]}
    elif expect == "zero":
        tail = vals[1:]
        same_sign = all(v > 0 for v in tail) or all(v < 0 for v in tail)
        # non-decreasing magnitude (tolerate flat/noisy steps) with NET growth
        # across the window -- a systematic, accumulating departure from zero,
        # not a one-step blip that converges back.
        non_decreasing = all(abs(vals[i + 1]) >= abs(vals[i]) - 1e-9
                             for i in range(len(vals) - 1))
        net_growth = abs(vals[-1]) > abs(vals[0])
        if same_sign and non_decreasing and net_growth:
            return {"failure": "drift", "window": window,
                    "from": vals[0], "to": vals[-1]}
    return None


def scan_progress(ledger: ProgressLedger, specs: dict, *, window: int = 3) -> list:
    """specs: {metric_name: 'decrease'|'zero'}.  Returns [(name, verdict), ...]
    for every monitored metric whose trend is currently violated."""
    out = []
    for name, expect in specs.items():
        s = ledger.series.get(name)
        if not s:
            continue
        v = detect_trend_failure(s, expect, window=window)
        if v:
            out.append((name, v))
    return out


def progress_protocol_block(name: str, verdict: dict) -> str:
    """LOUD self-triggered decompose-and-verify for a NON-CONVERGING trend -- the
    drift/stall the single-step surprise check is structurally blind to."""
    if verdict["failure"] == "drift":
        head = (f"[DEBUG — DRIFT] '{name}' has grown the SAME direction for "
                f"{verdict['window']} turns straight ({verdict['from']:.1f} -> "
                f"{verdict['to']:.1f}) — you are systematically VEERING OFF, not "
                f"converging.")
    else:
        head = (f"[DEBUG — NOT CONVERGING] '{name}' has NOT improved for "
                f"{verdict['window']} turns ({verdict['from']:.1f} -> "
                f"{verdict['to']:.1f}) although you keep acting toward the goal.")
    return (head + " Each single step looked fine (something changed) — which is "
            "exactly why a one-step polarity check MISSES this; the error is in "
            "the TREND. An assumption about how your action maps to its outcome is "
            "wrong. DEBUG it, don't push on: (1) decompose the action→effect chain "
            "(is the thing you track really the mover? is your control/aim "
            "computed from the RIGHT measured positions? does the effect's "
            "DIRECTION and MAGNITUDE match your prediction, not merely its "
            "polarity?). (2) Measure ONE step's predicted-vs-actual vector from "
            "raw pixels. (3) Localize the wrong assumption and CORRECT the mapping "
            "before spending more actions.")


def protocol_block(anomaly: Optional[dict] = None) -> str:
    """The LOUD decompose-and-verify block to surface when a surprise fires."""
    head = ("[DEBUG — SOMETHING DOESN'T ADD UP]")
    if anomaly:
        head += (f" You predicted a {anomaly['expected'].replace('_', '-')} "
                 f"outcome but the measured result was {anomaly['observed'].replace('_', '-')}"
                 + (f" (\"{anomaly['prediction']}\")" if anomaly.get('prediction') else "")
                 + ".")
    return (head + " This contradiction means an ASSUMPTION is wrong — and the "
            "weight of probability is that it is YOUR OWN pipeline (tracking, "
            "aim, row/col-or-tick convention, measurement), NOT that the game is "
            "impossible or has a hidden mechanic; suspect your side FIRST. Do NOT "
            "re-guess a new explanation and do NOT conclude from the summary or "
            "the settled frame alone. DEBUG it:\n"
            "  1. List the assumptions the failed step rests on (action executed? "
            "control set right? the entity you watched is the mover? did you "
            "inspect the ANIMATION frames, not just the settled frame? did you "
            "measure pixels, not trust a label?).\n"
            "  2. For each, run the MINIMAL check that isolates it — inspect the "
            "raw per-frame animation / exact pixels / panel state, varying ONE "
            "thing.\n"
            "  3. Localize the FALSE assumption, then re-conclude only from the "
            "verified pieces. (Same skill as debugging code, a car, or any system.)")
