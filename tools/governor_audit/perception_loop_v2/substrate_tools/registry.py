"""Registry for VLM-directed substrate tools.

A *substrate tool* is an on-demand, game-agnostic operation the COS VLM invokes
by name (the perception reply's `visual_queries` list).  The substrate MEASURES
or RENDERS; the VLM INTERPRETS.  This module is the contribution surface:

  - ONE consistent interface every tool implements (see ToolSpec / the handler
    signature below).
  - A REGISTRY for register / unregister / enable / disable / list / dispatch,
    so tools can be managed (e.g. turned off in strict mode) without code edits.
  - render_vocabulary() generates the prompt block from the registered tools, so
    adding a tool AUTOMATICALLY advertises it to the VLM — no prompt hand-edit.

To add a tool, drop a module in this package and decorate a handler with
@tool(...); importing the package self-registers it.  See
docs/CONTRIBUTING_substrate_tools.md.

THE INTERFACE (consistent across all tools):

    @tool(name="my_op", summary="...", usage='{"op":"my_op","id":"k","bbox":[...]}',
          params={"bbox": "what it means"}, category="perception/visual",
          renders_image=False)
    def my_op(ctx: ToolContext, *, bbox, **_) -> dict:
        # ctx.frame  -> (n_ticks, n_ticks, 3) uint8 logical frame (already loaded)
        # ctx.n_ticks, ctx.out_dir (write PNGs here), ctx.query_id
        return {"some_measurement": ...}        # JSON-serialisable; no 'op'/'id'

  - The handler is called with the VLM's query dict (minus op/tool/type/id) as
    **kwargs, so ALWAYS end the signature with **_ to tolerate extra fields.
  - Return a JSON-serialisable dict.  A rendering tool writes its PNG to
    ctx.out_dir and returns {"image": "<filename>", ...}.
  - NEVER interpret game meaning, branch on game id, or bake a game-tuned magic
    number (the Prime Directive).  Measure / render only.
  - The dispatcher wraps every call, so a raised exception becomes an
    {"error": ...} result — one bad request never crashes the loop.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np

from .frameutils import FrameLike, load_logical_frame


# -----------------------------------------------------------------------------
# Tool interface
# -----------------------------------------------------------------------------


@dataclass
class ToolContext:
    """Everything a tool handler is given.  The frame is ALREADY loaded into the
    logical tick grid (load this once per batch, not per tool)."""
    frame: np.ndarray            # (n_ticks, n_ticks, 3) uint8 logical frame
    n_ticks: int
    out_dir: Path                # rendering tools write PNGs here
    query_id: str                # the VLM's id for this query (image naming)
    anim_frames: Optional[list] = None  # the last action's animation sub-frames
    #   (list of (n_ticks,n_ticks,3) arrays, oldest->settled) when it animated,
    #   else None — for animation-inspection tools.


@dataclass
class ToolSpec:
    name: str                    # the `op` the VLM uses
    summary: str                 # one-line description (-> prompt vocabulary)
    usage: str                   # example invocation (-> prompt vocabulary)
    handler: Callable            # (ctx, **args) -> dict
    params: dict = field(default_factory=dict)   # {param: description} (-> doc)
    category: str = "perception/visual"
    renders_image: bool = False
    enabled: bool = True
    contributor: str = "core"


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------


class ToolRegistry:
    """Holds ToolSpecs; supports management + dispatch.  One module-level
    singleton (REGISTRY) is the default everything uses."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    # -- management --------------------------------------------------------
    def register(self, spec: ToolSpec, *, override: bool = False) -> None:
        if spec.name in self._tools and not override:
            raise ValueError(
                f"substrate tool {spec.name!r} already registered "
                f"(pass override=True to replace)")
        self._tools[spec.name] = spec

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def enable(self, name: str) -> bool:
        t = self._tools.get(name)
        if t:
            t.enabled = True
        return bool(t)

    def disable(self, name: str) -> bool:
        t = self._tools.get(name)
        if t:
            t.enabled = False
        return bool(t)

    def is_enabled(self, name: str) -> bool:
        t = self._tools.get(name)
        return bool(t and t.enabled)

    def list(self, *, category: Optional[str] = None,
             include_disabled: bool = True) -> list[ToolSpec]:
        out = [t for t in self._tools.values()
               if (include_disabled or t.enabled)
               and (category is None or t.category == category)]
        return sorted(out, key=lambda t: (t.category, t.name))

    def names(self, *, enabled_only: bool = True) -> tuple[str, ...]:
        return tuple(sorted(n for n, t in self._tools.items()
                            if t.enabled or not enabled_only))

    def apply_env_disables(self, env_var: str = "COS_DISABLED_TOOLS") -> None:
        """Disable tools named (comma-separated) in $COS_DISABLED_TOOLS — lets a
        run turn tools off without code edits (e.g. strict-mode trimming)."""
        for nm in (x.strip() for x in os.environ.get(env_var, "").split(",")):
            if nm:
                self.disable(nm)


REGISTRY = ToolRegistry()


def tool(*, name: str, summary: str, usage: str,
         params: Optional[dict] = None, category: str = "perception/visual",
         renders_image: bool = False, contributor: str = "core",
         registry: Optional[ToolRegistry] = None) -> Callable:
    """Decorator: register the decorated handler as a substrate tool."""
    reg = registry or REGISTRY

    def deco(fn: Callable) -> Callable:
        reg.register(ToolSpec(
            name=name, summary=summary, usage=usage, handler=fn,
            params=params or {}, category=category,
            renders_image=renders_image, contributor=contributor))
        return fn

    return deco


# -----------------------------------------------------------------------------
# Dispatch + prompt vocabulary
# -----------------------------------------------------------------------------


def run_queries(frame: FrameLike, queries: Sequence[dict],
                out_dir: Union[str, Path], *, n_ticks: int = 64,
                anim_frames: Optional[Sequence[FrameLike]] = None,
                registry: Optional[ToolRegistry] = None) -> list:
    """Fulfil a VLM-supplied list of tool queries against one frame.

    Each query is a dict carrying an `op` (or `tool`/`type`) plus that op's args
    and an optional `id`.  The frame is loaded once; each query is dispatched to
    its registered handler and wrapped so a bad/disabled/unknown request returns
    an {"error": ...} result instead of raising — the loop must never crash on a
    malformed request.  Returns a list of JSON-serialisable result dicts, each
    carrying back `op` + `id`.
    """
    reg = registry or REGISTRY
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fr = load_logical_frame(frame, n_ticks=n_ticks)
    anim = ([load_logical_frame(f, n_ticks=n_ticks) for f in anim_frames]
            if anim_frames else None)
    results: list = []
    for idx, q in enumerate(queries or []):
        if not isinstance(q, dict):
            results.append({"error": "query must be an object",
                            "raw": str(q), "id": f"q{idx}"})
            continue
        op = str(q.get("op") or q.get("tool") or q.get("type") or "").strip()
        qid = str(q.get("id", f"q{idx}"))
        spec = reg.get(op)
        if spec is None:
            results.append({"error": f"unknown op {op!r}",
                            "known_ops": list(reg.names()), "op": op, "id": qid})
            continue
        if not spec.enabled:
            results.append({"error": f"op {op!r} is disabled",
                            "op": op, "id": qid})
            continue
        args = {k: v for k, v in q.items()
                if k not in ("op", "tool", "type", "id")}
        ctx = ToolContext(frame=fr, n_ticks=n_ticks, out_dir=out_dir,
                          query_id=qid, anim_frames=anim)
        try:
            res = spec.handler(ctx, **args)
            if not isinstance(res, dict):
                res = {"result": res}
        except TypeError as e:        # bad / missing args for this handler
            res = {"error": f"op {op!r} bad or missing args: {e}"}
        except KeyError as e:
            res = {"error": f"op {op!r} missing required arg {e}"}
        except Exception as e:        # never let one bad query crash the batch
            res = {"error": f"op {op!r} failed: {e}"}
        res = dict(res)
        res.setdefault("op", op)
        res["id"] = qid
        results.append(res)
    return results


_VOCAB_HEADER = (
    "ON-DEMAND VISUAL TOOLS — ask the substrate to MEASURE what you can't "
    "eyeball.\n"
    "You are reliable at interpretation but NOT at a few mechanical visual "
    "operations: exact counting, precise alignment / collinearity, distance, "
    "reading tiny or NESTED detail (a thumbnail / glyph / sub-grid inside a "
    "cell), and exact colour.  For any of these, do NOT guess — DELEGATE the "
    "measurement to the substrate by including a top-level `\"visual_queries\"` "
    "array in your JSON reply.  The substrate fulfils them and shows you the "
    "answers (and magnified images), then you re-emit your perception using "
    "those facts.  The substrate MEASURES only; YOU interpret — it will never "
    "tell you what a region \"is\" or \"means\".\n\n"
    "  Available tools (each query is an object with an `op` plus its args; add "
    "an `id` to correlate the answer):"
)


def render_vocabulary(*, registry: Optional[ToolRegistry] = None,
                      enabled_only: bool = True, n_ticks: int = 64) -> str:
    """Generate the perception-prompt tool block from the REGISTERED tools, so a
    newly-contributed tool is advertised to the VLM automatically."""
    reg = registry or REGISTRY
    lines = [_VOCAB_HEADER]
    for t in reg.list(include_disabled=not enabled_only):
        if enabled_only and not t.enabled:
            continue
        lines.append(f"    - {t.name}: {t.summary}")
        lines.append(f"        e.g. {t.usage}")
    lines.append(
        f"  All coordinates are tick coordinates in [0, {n_ticks}], the same "
        f"space as bboxes/cells.  Requesting tools is OPTIONAL — use them when a "
        f"measurement would make your report more accurate (especially to read "
        f"nested/sub-tile structure precisely), and omit the field entirely "
        f"when you don't need help.")
    return "\n".join(lines)
