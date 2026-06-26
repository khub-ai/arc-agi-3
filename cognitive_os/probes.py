"""Domain-neutral experimental probe contracts.

Probe proposals describe what the agent wants to test.  Probe executors
decide whether and how a domain can run that proposal safely.  The core
contract is intentionally small so ARC games, simulators, and robotics
adapters can share it without smuggling in domain-specific mechanics.

Two surfaces are exposed:

* The dict-shaped wire/storage form (``cos_probe_proposal/v1``), validated
  by :func:`validate_probe_proposal`.  Existing call sites that emit
  proposals as dicts (e.g. ARC shadow validation artifacts) continue to
  work unchanged.

* The typed :class:`ProbeProposal` dataclass.  A typed builder + holder
  for new code; ``.to_dict()`` produces the wire-form mapping.  Adds
  first-class fields for value-of-information arithmetic that the dict
  form has historically lacked: ``expected_information_gain``,
  ``expected_cost``, ``expected_duration_ms``, ``risk_class``.  This
  matters more in robotics (where probes cost real time, energy, and
  risk) than in ARC, but ARC also has expensive probes (e.g. a full
  precondition-scan burns real budget) so the asymmetry just lands
  earlier in robotics.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


PROBE_PROPOSAL_SCHEMA = "cos_probe_proposal/v1"
PROBE_RESULT_SCHEMA = "cos_probe_result/v1"


# Recognized risk classes.  Adapters may use other values; these are the
# canonical labels the engine's explorer-bid logic understands.  "safe"
# means a probe that doesn't change durable world state (read-only or
# fully reversible).  "destructive" means a probe whose effects are
# externally visible and persist (an ARC click that can't be undone, a
# robot push that knocks something over).  "irreversible" is the worst
# tier: state changes that cannot be recovered in-session.
RISK_CLASS_SAFE         = "safe"
RISK_CLASS_DESTRUCTIVE  = "destructive"
RISK_CLASS_IRREVERSIBLE = "irreversible"
_KNOWN_RISK_CLASSES = frozenset({
    RISK_CLASS_SAFE,
    RISK_CLASS_DESTRUCTIVE,
    RISK_CLASS_IRREVERSIBLE,
})


class ProbeValidationError(ValueError):
    """Raised when a probe proposal is malformed or unsafe by policy."""


class ProbeExecutor(Protocol):
    """Executor boundary for a domain-specific probe runner."""

    def execute(self, proposal: Mapping[str, Any]) -> dict[str, Any]:
        """Run or reject a probe proposal and return a probe result."""


@dataclass(frozen=True)
class ProbeProposal:
    """Typed proposal for an experimental probe.

    The dict-shaped form (``cos_probe_proposal/v1``) remains the wire and
    storage format; this class is a typed builder + holder.  ``to_dict()``
    produces the mapping that :func:`validate_probe_proposal` expects.

    Required fields mirror the wire-form requirements:

    * ``domain`` — adapter-supplied domain tag (e.g. ``"arc-agi-3"``,
      ``"robotics-sim"``).
    * ``name`` — short human-readable name for this proposal.
    * ``actions`` — non-empty sequence of action mappings, each with at
      least a ``primitive`` key and optional ``parameters``.

    Optional fields useful for any probe:

    * ``rationale`` — short prose explaining why this probe is being
      proposed.
    * ``target_hypothesis_id`` — the hypothesis-store id this probe is
      designed to test, if any.
    * ``success_criterion`` — short prose describing what observation
      would count as the probe succeeding.
    * ``observation_ref`` — adapter-defined reference to the observation
      context the probe should run against (e.g. game/level for ARC,
      pose/scene for robotics).
    * ``expected_observation`` — adapter-defined expected delta shape.

    Cost / value-of-information fields (added 2026-04-27 in response to
    the robotics-developer review captured in
    ``SPEC_hypothesis_validation_loop.md`` §8.1 point 7):

    * ``expected_information_gain`` — float in [0, 1], rough estimate of
      the credence-delta the probe would yield on its target hypothesis.
    * ``expected_cost`` — adapter-defined non-negative cost (ARC: budget
      units; robotics: time / energy / wear units).
    * ``expected_duration_ms`` — wall-clock duration estimate (optional;
      mostly relevant for robotics real-time scheduling).
    * ``risk_class`` — one of ``"safe"``, ``"destructive"``, or
      ``"irreversible"``; see module-level constants.

    Safety:

    * ``safety`` — adapter-supplied policy mapping.  ``real_world_allowed``
      defaults to absent / False; setting it True requires
      :func:`validate_probe_proposal` to be called with
      ``allow_real_world=True``.
    """

    domain:                    str
    name:                      str
    actions:                   tuple = ()
    # Optional descriptive fields.
    rationale:                 str = ""
    target_hypothesis_id:      Optional[str] = None
    success_criterion:         str = ""
    observation_ref:           Optional[Mapping[str, Any]] = None
    expected_observation:      Optional[Mapping[str, Any]] = None
    # Cost / VoI estimates (2026-04-27).  Defaults of zero / None
    # preserve backward compatibility with proposals that don't supply
    # them; the explorer-bid logic that consumes them must tolerate
    # zero / missing values.
    expected_information_gain: float = 0.0
    expected_cost:             float = 0.0
    expected_duration_ms:      Optional[float] = None
    risk_class:                str = RISK_CLASS_SAFE
    # Safety policy mapping.
    safety:                    Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render to the wire-form mapping accepted by validate_probe_proposal."""
        d: dict[str, Any] = {
            "schema":  PROBE_PROPOSAL_SCHEMA,
            "domain":  self.domain,
            "name":    self.name,
            "actions": [dict(a) for a in self.actions],
            "safety":  dict(self.safety),
        }
        if self.rationale:
            d["rationale"] = self.rationale
        if self.target_hypothesis_id:
            d["target_hypothesis_id"] = self.target_hypothesis_id
        if self.success_criterion:
            d["success_criterion"] = self.success_criterion
        if self.observation_ref:
            d["observation_ref"] = dict(self.observation_ref)
        if self.expected_observation:
            d["expected_observation"] = dict(self.expected_observation)
        # Always emit the estimates block even when zero — explorer logic
        # may want to distinguish "estimate is 0" from "no estimate given"
        # for proposals that genuinely expect zero info gain.  Cheap to
        # carry and consistently shaped.
        d["estimates"] = {
            "expected_information_gain": float(self.expected_information_gain),
            "expected_cost":             float(self.expected_cost),
            "risk_class":                str(self.risk_class),
        }
        if self.expected_duration_ms is not None:
            d["estimates"]["expected_duration_ms"] = float(self.expected_duration_ms)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProbeProposal":
        """Build a typed ProbeProposal from a wire-form mapping.

        Tolerates proposals without the ``estimates`` block (older proposals
        from before 2026-04-27) by treating the cost-VoI fields as zero /
        defaults.
        """
        actions_raw = data.get("actions") or ()
        actions = tuple(dict(a) for a in actions_raw)
        estimates = data.get("estimates") or {}
        return cls(
            domain                    = str(data.get("domain") or ""),
            name                      = str(data.get("name") or ""),
            actions                   = actions,
            rationale                 = str(data.get("rationale") or ""),
            target_hypothesis_id      = data.get("target_hypothesis_id"),
            success_criterion         = str(data.get("success_criterion") or ""),
            observation_ref           = data.get("observation_ref"),
            expected_observation      = data.get("expected_observation"),
            expected_information_gain = float(estimates.get("expected_information_gain", 0.0) or 0.0),
            expected_cost             = float(estimates.get("expected_cost", 0.0) or 0.0),
            expected_duration_ms      = (float(estimates["expected_duration_ms"])
                                          if "expected_duration_ms" in estimates else None),
            risk_class                = str(estimates.get("risk_class", RISK_CLASS_SAFE) or RISK_CLASS_SAFE),
            safety                    = data.get("safety") or {},
        )


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def validate_probe_proposal(
    proposal: Mapping[str, Any],
    *,
    allow_real_world: bool = False,
) -> None:
    """Validate the shared probe proposal shape and default safety policy.

    The default policy rejects proposals that request real-world execution.
    Robotics adapters should first route proposals through a simulator or
    external safety gate, then call this with ``allow_real_world=True`` only
    after that separate gate has explicitly promoted the probe.
    """

    if not _is_mapping(proposal):
        raise ProbeValidationError("probe proposal must be a mapping")
    if proposal.get("schema") != PROBE_PROPOSAL_SCHEMA:
        raise ProbeValidationError(
            f"probe proposal schema must be {PROBE_PROPOSAL_SCHEMA!r}"
        )
    if not str(proposal.get("domain") or "").strip():
        raise ProbeValidationError("probe proposal must include a domain")
    if not str(proposal.get("name") or "").strip():
        raise ProbeValidationError("probe proposal must include a name")

    actions = proposal.get("actions")
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
        raise ProbeValidationError("probe proposal actions must be a sequence")
    if not actions:
        raise ProbeValidationError("probe proposal must include at least one action")
    for index, action in enumerate(actions, start=1):
        if not _is_mapping(action):
            raise ProbeValidationError(f"action {index} must be a mapping")
        if not str(action.get("primitive") or "").strip():
            raise ProbeValidationError(f"action {index} must include a primitive")
        parameters = action.get("parameters")
        if parameters is not None and not _is_mapping(parameters):
            raise ProbeValidationError(f"action {index} parameters must be a mapping")

    safety = proposal.get("safety")
    if safety is not None and not _is_mapping(safety):
        raise ProbeValidationError("probe proposal safety must be a mapping")
    if (
        _is_mapping(safety)
        and safety.get("real_world_allowed") is True
        and not allow_real_world
    ):
        raise ProbeValidationError(
            "real-world probe execution is not allowed by the default COS policy"
        )

    # Cost / value-of-information fields (added 2026-04-27, optional).
    # These live under "estimates" by convention; absent block is fine
    # (older proposals didn't carry them and the explorer must tolerate
    # zero / missing values).
    estimates = proposal.get("estimates")
    if estimates is not None:
        if not _is_mapping(estimates):
            raise ProbeValidationError("probe proposal estimates must be a mapping")
        for _k in ("expected_information_gain", "expected_cost"):
            if _k in estimates:
                _v = estimates[_k]
                if not isinstance(_v, (int, float)) or _v < 0:
                    raise ProbeValidationError(
                        f"estimates.{_k} must be a non-negative number"
                    )
        if "expected_duration_ms" in estimates:
            _d = estimates["expected_duration_ms"]
            if _d is not None and (not isinstance(_d, (int, float)) or _d < 0):
                raise ProbeValidationError(
                    "estimates.expected_duration_ms must be a non-negative number or None"
                )
        if "risk_class" in estimates:
            _rc = estimates["risk_class"]
            if not isinstance(_rc, str):
                raise ProbeValidationError("estimates.risk_class must be a string")


def make_probe_result(
    proposal: Mapping[str, Any],
    *,
    executor: str,
    status: str,
    observation_delta: Mapping[str, Any] | None = None,
    safety: Mapping[str, Any] | None = None,
    notes: Sequence[str] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a serializable ``cos_probe_result/v1`` object."""

    result: dict[str, Any] = {
        "schema": PROBE_RESULT_SCHEMA,
        "proposal_schema": proposal.get("schema"),
        "proposal_name": proposal.get("name"),
        "domain": proposal.get("domain"),
        "executor": executor,
        "status": status,
        "observation_delta": dict(observation_delta or {}),
        "safety": {
            "real_world_executed": False,
        },
        "notes": list(notes or []),
    }
    if safety:
        result["safety"].update(dict(safety))
    if error:
        result["error"] = error
    return result


class SimProbeExecutor:
    """Safe placeholder for domains that require a simulator.

    This executor does not simulate dynamics itself.  It exists so the
    substrate can carry the same proposal/result contract into robotics
    work without accidentally granting hardware authority.
    """

    def __init__(self, *, executor_name: str = "simulator-required") -> None:
        self.executor_name = executor_name

    def execute(self, proposal: Mapping[str, Any]) -> dict[str, Any]:
        validate_probe_proposal(proposal)
        return make_probe_result(
            proposal,
            executor=self.executor_name,
            status="sim_required",
            observation_delta={},
            safety={
                "real_world_executed": False,
                "simulated": False,
                "public_observation_only": True,
            },
            notes=[
                "No simulator adapter was supplied; proposal was not executed.",
            ],
        )
