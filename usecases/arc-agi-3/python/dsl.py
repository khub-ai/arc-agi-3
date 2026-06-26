"""Probe DSL parser + validator.

Grammar v0.1 (closed):

INSTRUCTIONS
  DO <ACTION_LABEL>
  DO_SEQ [<ACTION_LABEL>, ...]
  REPEAT DO <ACTION_LABEL> <N>              # 1 <= N <= 10
  RESET

OBSERVATIONS
  REGION_DELTA [r0,c0,r1,c1]
  ELEMENT_MOVED <element_id>
  STATE
  AVAILABLE_ACTIONS
  SCORE_DELTA
  CHANGE_REPORT
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union


ACTION_RE     = re.compile(r"^ACTION[0-9]+$")
INT_RE        = re.compile(r"^-?\d+$")
BBOX_RE       = re.compile(r"^\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]$")
LIST_ACT_RE   = re.compile(r"^\[\s*([A-Z0-9_,\s]+)\s*\]$")


@dataclass
class DoOne:
    action: str


@dataclass
class DoSeq:
    actions: List[str]


@dataclass
class RepeatDo:
    action: str
    n:      int


@dataclass
class Reset:
    pass


Instruction = Union[DoOne, DoSeq, RepeatDo, Reset]


@dataclass
class ObsRegionDelta:
    bbox: Tuple[int, int, int, int]


@dataclass
class ObsElementMoved:
    element_id: int


@dataclass
class ObsState:
    pass


@dataclass
class ObsAvailableActions:
    pass


@dataclass
class ObsScoreDelta:
    pass


@dataclass
class ObsChangeReport:
    pass


Observation = Union[
    ObsRegionDelta, ObsElementMoved, ObsState, ObsAvailableActions,
    ObsScoreDelta, ObsChangeReport,
]


class DSLError(ValueError):
    pass


def parse_instruction(text: str, available_actions: List[str]) -> Instruction:
    t = text.strip()
    if t == "RESET":
        return Reset()
    if t.startswith("DO_SEQ"):
        rest = t[len("DO_SEQ"):].strip()
        m = LIST_ACT_RE.match(rest)
        if not m:
            raise DSLError(f"DO_SEQ expects [ACT,ACT,...]: got {text!r}")
        items = [x.strip() for x in m.group(1).split(",") if x.strip()]
        for a in items:
            _require_action(a, available_actions)
        if not items:
            raise DSLError(f"DO_SEQ requires >= 1 action: {text!r}")
        return DoSeq(items)
    if t.startswith("REPEAT DO"):
        rest = t[len("REPEAT DO"):].strip().split()
        if len(rest) != 2:
            raise DSLError(f"REPEAT DO requires <ACTION> <N>: got {text!r}")
        act, n_str = rest
        _require_action(act, available_actions)
        if not INT_RE.match(n_str):
            raise DSLError(f"REPEAT DO N must be integer: got {n_str!r}")
        n = int(n_str)
        if not 1 <= n <= 10:
            raise DSLError(f"REPEAT DO N must be in 1..10: got {n}")
        return RepeatDo(act, n)
    if t.startswith("DO "):
        act = t[3:].strip()
        _require_action(act, available_actions)
        return DoOne(act)
    raise DSLError(f"unknown instruction: {text!r}")


def parse_observation(
    text:               str,
    valid_element_ids:  set[int],
    frame_shape:        Tuple[int, int],
) -> Observation:
    t = text.strip()
    if t == "STATE":
        return ObsState()
    if t == "AVAILABLE_ACTIONS":
        return ObsAvailableActions()
    if t == "SCORE_DELTA":
        return ObsScoreDelta()
    if t == "CHANGE_REPORT":
        return ObsChangeReport()
    if t.startswith("REGION_DELTA"):
        rest = t[len("REGION_DELTA"):].strip()
        m = BBOX_RE.match(rest)
        if not m:
            raise DSLError(f"REGION_DELTA expects [r0,c0,r1,c1]: got {text!r}")
        r0, c0, r1, c1 = (int(m.group(i)) for i in (1, 2, 3, 4))
        h, w = frame_shape
        if not (0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w):
            raise DSLError(f"REGION_DELTA bbox out of bounds for {h}x{w}: {text!r}")
        return ObsRegionDelta((r0, c0, r1, c1))
    if t.startswith("ELEMENT_MOVED"):
        rest = t[len("ELEMENT_MOVED"):].strip()
        if not INT_RE.match(rest):
            raise DSLError(f"ELEMENT_MOVED expects integer element id: got {text!r}")
        eid = int(rest)
        if eid not in valid_element_ids:
            raise DSLError(
                f"ELEMENT_MOVED references unknown element id {eid}; "
                f"valid={sorted(valid_element_ids)}"
            )
        return ObsElementMoved(eid)
    raise DSLError(f"unknown observation: {text!r}")


def _require_action(label: str, available: List[str]) -> None:
    if not ACTION_RE.match(label):
        raise DSLError(f"action label malformed: {label!r}")
    if label not in available:
        raise DSLError(
            f"action {label!r} not in AVAILABLE_ACTIONS {available}"
        )


@dataclass
class ProbeParseResult:
    probe_id:        str
    hypothesis:      str
    instructions:    List[Instruction]
    observations:    List[Observation]
    outcome_map:     dict
    errors:          List[str]   # empty => valid


def parse_probe(
    probe_json:         dict,
    available_actions:  List[str],
    valid_element_ids:  set[int],
    frame_shape:        Tuple[int, int],
) -> ProbeParseResult:
    errors: List[str] = []
    instructions: List[Instruction] = []
    observations: List[Observation] = []

    pid = probe_json.get("probe_id", "?")
    hyp = probe_json.get("hypothesis", "")

    for raw in probe_json.get("instructions", []) or []:
        try:
            instructions.append(parse_instruction(str(raw), available_actions))
        except DSLError as e:
            errors.append(f"instruction {raw!r}: {e}")

    for raw in probe_json.get("observe", []) or []:
        try:
            observations.append(
                parse_observation(str(raw), valid_element_ids, frame_shape)
            )
        except DSLError as e:
            errors.append(f"observation {raw!r}: {e}")

    if not instructions and not observations:
        errors.append("empty probe: no instructions, no observations")

    return ProbeParseResult(
        probe_id     = pid,
        hypothesis   = hyp,
        instructions = instructions,
        observations = observations,
        outcome_map  = probe_json.get("outcome_map", {}) or {},
        errors       = errors,
    )
