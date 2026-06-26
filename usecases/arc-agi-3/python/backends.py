"""Backends for Oracle (OpenAI / Anthropic) and PUPIL (OpenRouter).

Both return a raw text reply -- caller handles JSON parsing.  Keys are
loaded from .env at KF repo root or .env.local,
matching the convention used by other KF usecases.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import http.client
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# On-disk call cache
# ---------------------------------------------------------------------------
#
# Every call is keyed on sha256(model + system + user + image_b64 +
# max_tokens + temperature) so identical prompts are never re-billed.
# Cache is persistent across sessions; delete the dir to force refresh.

CACHE_DIR = Path(__file__).resolve().parents[1] / ".tmp" / "model_cache"


def _cache_key(
    model: str, system: str, user: str,
    image_b64: Optional[str], max_tokens: int, temperature: float,
) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x1f")
    h.update(system.encode("utf-8"))
    h.update(b"\x1f")
    h.update(user.encode("utf-8"))
    h.update(b"\x1f")
    h.update((image_b64 or "").encode("ascii"))
    h.update(b"\x1f")
    h.update(f"{max_tokens}:{temperature}".encode("ascii"))
    return h.hexdigest()


def _cache_get(key: str) -> Optional[dict]:
    p = CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if (
            obj.get("model", "").startswith("gpt-")
            and not obj.get("reply")
            and isinstance(obj.get("raw"), dict)
        ):
            obj["reply"] = _extract_openai_output_text(obj["raw"])
        obj["_cache_hit"] = True
        return obj
    except Exception:  # noqa: BLE001
        return None


def _cache_put(key: str, entry: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    to_save = {k: v for k, v in entry.items() if k != "_cache_hit"}
    (CACHE_DIR / f"{key}.json").write_text(
        json.dumps(to_save, indent=2), encoding="utf-8"
    )


def _cached_call(
    *,
    model: str, system: str, user: str,
    image_b64: Optional[str], max_tokens: int, temperature: float,
    uncached_fn: Callable[[], dict],
) -> dict:
    key = _cache_key(model, system, user, image_b64, max_tokens, temperature)
    hit = _cache_get(key)
    # An empty cached reply is a poisoned entry — typically the
    # provider returned an empty body once (rate-limit, content-
    # filter, transient outage), the cache stored that, and every
    # subsequent identical-prompt call returns the same empty body
    # forever.  Treat empty cache hits as a miss so the fresh call
    # has a chance to succeed.  See trials trial_3of7_postfix_g31{f,g,h,i}
    # which all died at the SAME turn for this exact reason.
    if hit is not None:
        if (hit.get("reply") or "").strip():
            return hit
    result = uncached_fn()
    result["_cache_key"] = key
    result["_cache_hit"] = False
    # Only cache non-empty replies — same rationale as above.
    if (result.get("reply") or "").strip():
        _cache_put(key, result)
    return result


# ---------------------------------------------------------------------------
# Key loading
# ---------------------------------------------------------------------------

KEY_FILES = [
    Path(r".env.local"),
    Path(r".env.local"),
]


def _load_keys() -> None:
    for p in KEY_FILES:
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and not os.environ.get(k):
                os.environ[k] = v


_load_keys()


# ---------------------------------------------------------------------------
# OpenAI (Oracle: gpt-5.4)
# ---------------------------------------------------------------------------

def call_openai(
    *,
    model:        str,
    system:       str,
    user:         str,
    image_b64:    Optional[str]  = None,
    max_tokens:   int            = 4000,
    temperature:  float          = 0.0,
    timeout_s:    int            = 300,
) -> dict:
    return _cached_call(
        model=model, system=system, user=user, image_b64=image_b64,
        max_tokens=max_tokens, temperature=temperature,
        uncached_fn=lambda: _call_openai_uncached(
            model=model, system=system, user=user, image_b64=image_b64,
            max_tokens=max_tokens, temperature=temperature,
            timeout_s=timeout_s,
        ),
    )


def _call_openai_uncached(
    *,
    model:        str,
    system:       str,
    user:         str,
    image_b64:    Optional[str]  = None,
    max_tokens:   int            = 4000,
    temperature:  float          = 0.0,
    timeout_s:    int            = 300,
) -> dict:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")

    content: list[dict] = [{"type": "input_text", "text": user}]
    if image_b64:
        content.append({
            "type": "input_image",
            "image_url": f"data:image/png;base64,{image_b64}",
        })

    body = {
        "model":             model,
        "instructions":      system,
        "input":             [{"role": "user", "content": content}],
        "max_output_tokens": max_tokens,
        "temperature":       temperature,
    }
    # Reasoning models reject `temperature`: o1*, o3*, o4*, and *-pro variants.
    if (
        model.startswith(("o1", "o3", "o4"))
        or "-pro" in model
    ):
        body.pop("temperature", None)
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )

    last_err: Exception | None = None
    for attempt in range(3):
        if attempt:
            time.sleep(15 * attempt)
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                resp = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            body_txt = e.read().decode("utf-8", "replace")
            if e.code in (408, 429, 500, 502, 503, 504) and attempt < 2:
                last_err = RuntimeError(f"openai HTTP {e.code}: {body_txt}")
                continue
            raise RuntimeError(f"openai HTTP {e.code}: {body_txt}") from e
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last_err = e
            if attempt < 2:
                print(f"  [backends] OpenAI timeout/error (attempt {attempt+1}/3): {e}")
                continue
            raise
    else:
        raise RuntimeError(f"OpenAI call failed after 3 attempts: {last_err}") from last_err

    elapsed = time.time() - t0

    usage = resp.get("usage") or {}
    input_tokens  = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    # gpt-5.4: $2.50/M input, $15/M output
    cost_usd = (input_tokens * 2.5 + output_tokens * 15) / 1_000_000
    return {
        "model":         model,
        "reply":         _extract_openai_output_text(resp),
        "latency_ms":    int(elapsed * 1000),
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "cost_usd":      round(cost_usd, 6),
        "provider":      "openai",
        "raw":           resp,
    }


def _extract_openai_output_text(resp: dict) -> str:
    text = resp.get("output_text")
    if isinstance(text, str) and text:
        return text

    parts: list[str] = []
    for item in resp.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") == "output_text":
                chunk = content.get("text")
                if isinstance(chunk, str) and chunk:
                    parts.append(chunk)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Anthropic (Oracle: claude-sonnet-4-6)
# ---------------------------------------------------------------------------

def call_anthropic(
    *,
    model:        str,
    system:       str,
    user:         str,
    image_b64:    Optional[str]  = None,
    max_tokens:   int            = 4000,
    temperature:  float          = 0.0,
    timeout_s:    int            = 300,
) -> dict:
    return _cached_call(
        model=model, system=system, user=user, image_b64=image_b64,
        max_tokens=max_tokens, temperature=temperature,
        uncached_fn=lambda: _call_anthropic_uncached(
            model=model, system=system, user=user, image_b64=image_b64,
            max_tokens=max_tokens, temperature=temperature,
            timeout_s=timeout_s,
        ),
    )


def _call_anthropic_uncached(
    *,
    model:        str,
    system:       str,
    user:         str,
    image_b64:    Optional[str]  = None,
    max_tokens:   int            = 4000,
    temperature:  float          = 0.0,
    timeout_s:    int            = 300,
) -> dict:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    content: list = []
    if image_b64:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image_b64,
            },
        })
    content.append({"type": "text", "text": user})

    body = {
        "model":       model,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        # Wrap system in a list so cache_control can be attached.
        # Anthropic caches the system prefix server-side for 5 min;
        # turns 2+ in a session pay $0.30/M instead of $3/M (~90% savings).
        "system": [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages":    [{"role": "user", "content": content}],
    }
    # claude-opus-4-* rejects the temperature param ("deprecated for this model").
    if model.startswith("claude-opus-4"):
        body.pop("temperature", None)
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    last_err: Exception | None = None
    for attempt in range(3):
        if attempt:
            time.sleep(15 * attempt)
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                resp = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            body_txt = e.read().decode("utf-8", "replace")
            if e.code in (429, 529) and attempt < 2:
                last_err = RuntimeError(f"anthropic HTTP {e.code}: {body_txt}")
                continue
            raise RuntimeError(f"anthropic HTTP {e.code}: {body_txt}") from e
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last_err = e
            if attempt < 2:
                print(f"  [backends] Anthropic timeout/error (attempt {attempt+1}/3): {e}")
                continue
            raise
    else:
        raise RuntimeError(f"Anthropic call failed after 3 attempts: {last_err}") from last_err

    elapsed = time.time() - t0

    text_parts = [c.get("text", "") for c in resp.get("content", []) if c.get("type") == "text"]
    usage = resp.get("usage") or {}
    input_tokens        = int(usage.get("input_tokens",             0))
    output_tokens       = int(usage.get("output_tokens",            0))
    cache_write_tokens  = int(usage.get("cache_creation_input_tokens", 0))
    cache_read_tokens   = int(usage.get("cache_read_input_tokens",     0))
    # claude-sonnet-4-6: $3/M input, $15/M output
    # with prompt caching: $3.75/M write, $0.30/M read (vs $3/M uncached)
    uncached_input = input_tokens - cache_write_tokens - cache_read_tokens
    cost_usd = (
        uncached_input    * 3.00
        + cache_write_tokens * 3.75
        + cache_read_tokens  * 0.30
        + output_tokens      * 15.0
    ) / 1_000_000
    return {
        "model":               model,
        "reply":               "".join(text_parts),
        "latency_ms":          int(elapsed * 1000),
        "input_tokens":        input_tokens,
        "output_tokens":       output_tokens,
        "cache_write_tokens":  cache_write_tokens,
        "cache_read_tokens":   cache_read_tokens,
        "cost_usd":            round(cost_usd, 6),
        "provider":            "anthropic",
        "raw":                 resp,
    }


def _normalize_openrouter_model(model: str) -> str:
    # Accept bare shortcuts like "gemini-2.5-pro" and map to vendor slug.
    if "/" in model:
        return model
    if model.startswith("gemini-"):
        return f"google/{model}"
    return model


def _normalize_gemma_together_slug(model: str) -> str:
    """Map google/gemma-*-it slugs to Together's casing convention.

    Together's catalog uses capital B in size suffixes
    (``google/gemma-4-31B-it``, ``google/gemma-4-26B-A4B-it``); the
    OpenRouter slug for the same model uses lowercase b
    (``google/gemma-4-31b-it``).  Both are seen in callers and configs;
    accept either and emit the Together form.
    """
    import re
    # Replace ``-NNb-`` (case-insensitive) with ``-NNB-`` to match
    # Together's slugs.  Leaves the rest of the slug untouched.
    return re.sub(
        r"-(\d+)b(-|$)",
        lambda m: f"-{m.group(1)}B{m.group(2)}",
        model,
    )


# ---------------------------------------------------------------------------
# API SPEND tracking + budget cap (ArcPrize track: $10K hard limit).
# Every backend returns an accurate per-call ``cost_usd`` (OpenRouter's exact
# billed cost, else the per-model table; OpenAI/Anthropic compute it too).  We
# accumulate it into a ledger under COS_KB_ROOT so the total persists ACROSS the
# competition's games (COS_KB_ROOT is the per-session writable root), and refuse
# a call once the cap is reached so spend never crosses it.  Local/Ollama calls
# never reach call_oracle, so the local path stays free + uncapped.  Opt-in:
# COS_COST_CAP_USD unset/0 => unlimited (dev + local); set it for the API track.
# ---------------------------------------------------------------------------
class BudgetExceeded(RuntimeError):
    """Raised by check_budget()/call_oracle when the API spend cap is reached."""


_SPEND = {"usd": None}   # cumulative USD this session; lazy-loaded from the ledger


def _cost_ledger_path() -> Optional[Path]:
    root = os.environ.get("COS_KB_ROOT")
    return (Path(root) / "cost_ledger.json") if root else None


def get_spend() -> float:
    """Cumulative API USD spent so far this session (loaded from the ledger once)."""
    if _SPEND["usd"] is None:
        p = _cost_ledger_path()
        try:
            _SPEND["usd"] = float(json.loads(p.read_text())["usd"]) if (p and p.exists()) else 0.0
        except Exception:
            _SPEND["usd"] = 0.0
    return _SPEND["usd"]


def record_spend(cost_usd) -> float:
    """Add one call's cost to the cumulative total and persist it.  Returns the new total."""
    total = get_spend() + max(0.0, float(cost_usd or 0.0))
    _SPEND["usd"] = total
    p = _cost_ledger_path()
    if p:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"usd": round(total, 6)}))
        except Exception:
            pass
    return total


def cost_cap_usd() -> float:
    """Configured hard cap (COS_COST_CAP_USD); 0/unset => unlimited."""
    try:
        return float(os.environ.get("COS_COST_CAP_USD", "0") or 0.0)
    except Exception:
        return 0.0


def check_budget() -> None:
    """Raise BudgetExceeded if the cap is set and the cumulative spend has reached it."""
    cap = cost_cap_usd()
    if cap and get_spend() >= cap:
        raise BudgetExceeded(
            f"API budget cap reached: ${get_spend():.2f} spent >= ${cap:.2f} cap")


def call_oracle(
    *,
    model:        str,
    system:       str,
    user:         str,
    image_b64:    Optional[str]  = None,
    max_tokens:   int            = 4000,
    temperature:  float          = 0.0,
    timeout_s:    int            = 300,
) -> dict:
    # Human-in-the-loop backend: model="human:<label>" or just
    # "human" routes to the file-based handoff.  The harness sets
    # VLM_HUMAN_PENDING_DIR; this backend writes prompt + image
    # there and polls for a reply JSON file.  Used for sitting an
    # operator/external LLM in the VLM seat without modifying the
    # rest of the system.
    if model == "human" or model.startswith("human:"):
        return call_human_vlm(
            model=model, system=system, user=user,
            image_b64=image_b64,
        )
    # Local self-hosted (Ollama / any OpenAI-compatible) endpoint. Slug form:
    #   ollama/<host:port>/<model_tag>   e.g. ollama/127.0.0.1:11434/qwen3-vl:8b
    # or  ollama/<model_tag>  with host from COS_OLLAMA_HOST. No API key, no
    # internet — this is the actual OFFLINE self-host path.
    if model.startswith("ollama/") or model.startswith("local/"):
        return call_ollama(
            model=model, system=system, user=user, image_b64=image_b64,
            max_tokens=max_tokens, temperature=temperature, timeout_s=timeout_s,
        )
    if model.startswith("gpt-"):
        return call_openai(
            model=model, system=system, user=user, image_b64=image_b64,
            max_tokens=max_tokens, temperature=temperature, timeout_s=timeout_s,
        )
    if model.startswith("claude-"):
        return call_anthropic(
            model=model, system=system, user=user, image_b64=image_b64,
            max_tokens=max_tokens, temperature=temperature, timeout_s=timeout_s,
        )
    # Together.ai serves "deepseek-ai/..." (the canonical slug).  We prefer
    # this over OpenRouter for these models because the OpenRouter→Together
    # shared key is heavily rate-limited; a direct TOGETHER_API_KEY
    # bypasses that cap entirely.
    if model.startswith("deepseek-ai/") and os.environ.get("TOGETHER_API_KEY"):
        return call_together(
            model=model, system=system, user=user, image_b64=image_b64,
            max_tokens=max_tokens, temperature=temperature,
        )
    # google/gemma-* -> Together when TOGETHER_API_KEY is set AND
    # there's no image in the request.  Together hosts Gemma 4 with
    # lower latency than OpenRouter (~3-5s vs ~80s) and avoids the
    # OpenRouter→Together shared-key throttle.  Slug case is
    # normalised here (Together uses capital B in size suffix; e.g.
    # gemma-4-31B-it) so callers can pass either form.
    #
    # Multimodal restriction: Together's Gemma deployments as of
    # 2026-05-07 are text-only FP8 quantizations (verified via
    # /v1/models — all Gemma variants report image_pixel=0 in
    # pricing and 500-error on multimodal input).  When image_b64 is
    # present we fall through to OpenRouter, whose Gemma serves
    # multimodal correctly.  See trial trial_together_gemma turn 3:
    # bad JSON from Oracle: no JSON in reply — model produced empty
    # content because vision input wasn't accepted.
    if (model.startswith("google/gemma-")
            and os.environ.get("TOGETHER_API_KEY")
            and image_b64 is None):
        return call_together(
            model=_normalize_gemma_together_slug(model),
            system=system, user=user, image_b64=None,
            max_tokens=max_tokens, temperature=temperature,
        )
    # gemini / qwen / any vendor-slug model -> OpenRouter
    return call_openrouter(
        model=_normalize_openrouter_model(model),
        system=system, user=user, image_b64=image_b64,
        max_tokens=max_tokens, temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Together.ai (direct, bypasses OpenRouter rate-limit on shared keys)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Local self-hosted (Ollama / OpenAI-compatible) — the OFFLINE self-host path
# ---------------------------------------------------------------------------

def call_ollama(
    *,
    model:        str,
    system:       str,
    user:         str,
    image_b64:    Optional[str]  = None,
    max_tokens:   int            = 4000,
    temperature:  float          = 0.0,
    timeout_s:    int            = 600,
) -> dict:
    return _cached_call(
        model=model, system=system, user=user, image_b64=image_b64,
        max_tokens=max_tokens, temperature=temperature,
        uncached_fn=lambda: _call_ollama_uncached(
            model=model, system=system, user=user, image_b64=image_b64,
            max_tokens=max_tokens, temperature=temperature, timeout_s=timeout_s,
        ),
    )


def _call_ollama_uncached(
    *,
    model:        str,
    system:       str,
    user:         str,
    image_b64:    Optional[str]  = None,
    max_tokens:   int            = 4000,
    temperature:  float          = 0.0,
    timeout_s:    int            = 600,
) -> dict:
    # Parse "ollama/<host:port>/<tag>" or "ollama/<tag>" (host from env).  The TAG
    # selects the CoT mode: 'qwen3-vl:8b' always emits a reasoning trace; the
    # 'qwen3-vl:8b-instruct' tag does not.  That -- not any API flag -- is how
    # thinking is controlled: Ollama's `think` param and the `/no_think` prompt
    # directive are BOTH no-ops on the vl renderer (and `think:false` slightly
    # degrades output); see DEPLOYMENT_GUIDE section 8b.
    rest = model.split("/", 1)[1]
    if "/" in rest:
        host, model_tag = rest.split("/", 1)
    else:
        model_tag = rest
        host = os.environ.get("COS_OLLAMA_HOST", "127.0.0.1:11434")
    for sfx in (":think", ":nothink"):       # legacy selectors -> ignored now
        if model_tag.endswith(sfx):
            model_tag = model_tag[: -len(sfx)]

    user_msg: dict = {"role": "user", "content": user}
    if image_b64:
        user_msg["images"] = [image_b64]   # native /api/chat: raw base64 list

    # num_gpu=99 forces every layer onto the GPU (without it the runner falls into
    # a CPU/GPU hybrid split capped near 15% GPU on the reference 1080 Ti); num_ctx
    # 8192 covers a COS turn (frame + history + any thinking trace + answer);
    # num_predict=-1 lets a thinking trace finish (it is a SEPARATE parser field
    # from content, so it never starves the answer).  Do NOT pass `think`.
    body = {
        "model":    model_tag,
        "messages": [{"role": "system", "content": system}, user_msg],
        "stream":   False,
        "options":  {
            "temperature": temperature,
            "top_p":       0.95,
            "num_predict": -1,
            "num_ctx":     8192,
            "num_gpu":     99,
        },
    }
    req = urllib.request.Request(
        f"http://{host}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"ollama HTTP {e.code}: {e.read().decode('utf-8','replace')}"
        ) from e
    elapsed = time.time() - t0

    msg = resp.get("message") or {}
    text = msg.get("content") or ""
    if not text:
        text = msg.get("thinking") or ""

    return {
        "text":          text,
        "reply":         text,
        "input_tokens":  int(resp.get("prompt_eval_count", 0) or 0),
        "output_tokens": int(resp.get("eval_count", 0) or 0),
        "cost_usd":      0.0,
        "latency_ms":    int(elapsed * 1000),
        "model":         model,
        "provider":      "ollama",
        "raw":           resp,
    }


def call_together(
    *,
    model:        str,
    system:       str,
    user:         str,
    image_b64:    Optional[str]  = None,
    max_tokens:   int            = 4000,
    temperature:  float          = 0.0,
) -> dict:
    return _cached_call(
        model=model, system=system, user=user, image_b64=image_b64,
        max_tokens=max_tokens, temperature=temperature,
        uncached_fn=lambda: _call_together_uncached(
            model=model, system=system, user=user, image_b64=image_b64,
            max_tokens=max_tokens, temperature=temperature,
        ),
    )


def _call_together_uncached(
    *,
    model:        str,
    system:       str,
    user:         str,
    image_b64:    Optional[str]  = None,
    max_tokens:   int            = 4000,
    temperature:  float          = 0.0,
) -> dict:
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        raise RuntimeError("TOGETHER_API_KEY not set")

    user_content: list = []
    if image_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
        })
    user_content.append({"type": "text", "text": user})

    body = {
        "model":       model,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ],
    }
    req = urllib.request.Request(
        "https://api.together.xyz/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {key}",
            # Together's edge (Cloudflare) blocks bare urllib UA strings
            # with code 1010; explicit UA gets through.
            "User-Agent":    "Mozilla/5.0 (compat; arc-bench/1.0)",
        },
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"together HTTP {e.code}: {e.read().decode('utf-8','replace')}"
        ) from e
    elapsed = time.time() - t0

    text = ""
    if resp.get("choices"):
        msg = resp["choices"][0].get("message") or {}
        text = msg.get("content") or ""
        # Reasoning models (e.g. DeepSeek-V4, MiniMax) may leave content null
        # and put the answer in reasoning. Fall back so the caller gets
        # *something* — mirrors the OpenRouter path below.
        if not text:
            text = msg.get("reasoning") or ""

    usage = resp.get("usage") or {}
    in_tok  = int(usage.get("prompt_tokens", 0) or 0)
    out_tok = int(usage.get("completion_tokens", 0) or 0)
    # Together publishes per-model pricing on their site; without the live
    # rate card we conservatively report cost as 0 and let the caller
    # backfill from a static table if needed.
    cost_usd = 0.0

    return {
        "text":           text,
        "reply":          text,   # alias for callers expecting OpenAI/Anthropic shape
        "input_tokens":   in_tok,
        "output_tokens":  out_tok,
        "cost_usd":       cost_usd,
        "latency_ms":     int(elapsed * 1000),
        "model":          model,
        "provider":       "together.ai",
        "raw":            resp,
    }


# ---------------------------------------------------------------------------
# OpenRouter (PUPIL: google/gemma-4-26b-a4b-it)
# ---------------------------------------------------------------------------

def call_openrouter(
    *,
    model:        str,
    system:       str,
    user:         str,
    image_b64:    Optional[str]  = None,
    max_tokens:   int            = 4000,
    temperature:  float          = 0.0,
) -> dict:
    return _cached_call(
        model=model, system=system, user=user, image_b64=image_b64,
        max_tokens=max_tokens, temperature=temperature,
        uncached_fn=lambda: _call_openrouter_uncached(
            model=model, system=system, user=user, image_b64=image_b64,
            max_tokens=max_tokens, temperature=temperature,
        ),
    )


def _call_openrouter_uncached(
    *,
    model:        str,
    system:       str,
    user:         str,
    image_b64:    Optional[str]  = None,
    max_tokens:   int            = 4000,
    temperature:  float          = 0.0,
) -> dict:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    user_content: list = []
    if image_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
        })
    user_content.append({"type": "text", "text": user})

    body = {
        "model":       model,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ],
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {key}",
            "HTTP-Referer":  "https://github.com/khub-ai/arc-agi-3",
            "X-Title":       "ARC-AGI-3 Dialogic Distillation",
        },
        method="POST",
    )
    t0 = time.time()
    # Retry on upstream 429 (provider rate-limit on shared OpenRouter key).
    # Also retries on 5xx (provider transient outage).  Honors
    # retry_after_seconds when present in the error body, else backs
    # off exponentially.  Max 8 attempts with cap of 30s per wait —
    # ~2 minutes total worst case.  Trials lasting 5+ minutes
    # frequently hit a single multi-second rate window in the middle;
    # the prior 4-attempt / ~14s loop ran out and crashed the entire
    # session (e.g. trial_3of7_postfix_g31m).  Eight attempts with a
    # 30s cap covers the typical Parasail/Novita upstream-cap windows
    # without burning indefinite wall time on a sustained outage.
    attempt = 0
    _MAX_ATTEMPTS = 8
    while True:
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                _raw_body = r.read()
            resp = json.loads(_raw_body)
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            retryable = e.code == 429 or 500 <= e.code < 600
            if retryable and attempt < _MAX_ATTEMPTS:
                wait = min(30, 2 ** attempt)
                try:
                    err_meta = (json.loads(body).get("error", {})
                                .get("metadata", {}))
                    suggested = int(err_meta.get(
                        "retry_after_seconds", 0,
                    ) or 0)
                    if suggested > 0:
                        wait = min(60, max(wait, suggested + 1))
                except Exception:
                    pass
                print(f"  [openrouter {e.code}] retrying in {wait}s "
                      f"(attempt {attempt + 1}/{_MAX_ATTEMPTS})")
                time.sleep(wait)
                attempt += 1
                continue
            raise RuntimeError(f"openrouter HTTP {e.code}: {body}") from e
        except (json.JSONDecodeError, urllib.error.URLError,
                TimeoutError, http.client.IncompleteRead) as e:
            # Malformed-body / network-level failures.  Sometimes
            # OpenRouter returns a 200 with a truncated SSE stream or
            # a chunk mid-stream — json.loads then fails despite the
            # HTTP layer succeeding.  Treat the same as a 5xx: retry
            # with backoff; surface as RuntimeError if exhausted.
            if attempt < _MAX_ATTEMPTS:
                wait = min(30, 2 ** attempt)
                print(f"  [openrouter body-error] {type(e).__name__}: "
                      f"{e!s}; retrying in {wait}s "
                      f"(attempt {attempt + 1}/{_MAX_ATTEMPTS})")
                time.sleep(wait)
                attempt += 1
                continue
            raise RuntimeError(
                f"openrouter body-decode failed after "
                f"{_MAX_ATTEMPTS} attempts: {type(e).__name__}: {e!s}"
            ) from e
    elapsed = time.time() - t0

    text = ""
    if resp.get("choices"):
        msg = resp["choices"][0].get("message") or {}
        text = msg.get("content") or ""
        # Reasoning models (e.g. gemini-2.5-pro) may leave content null and
        # put the answer in reasoning. Fall back so Oracle gets *something*.
        if not text:
            text = msg.get("reasoning") or ""
    usage = resp.get("usage") or {}
    input_tokens  = int(usage.get("prompt_tokens",     0) or 0)
    output_tokens = int(usage.get("completion_tokens", 0) or 0)
    # Prefer OpenRouter's exact billed cost when provided.
    cost_usd = usage.get("cost")
    if cost_usd is None:
        pricing = {
            "google/gemini-2.5-pro":           (1.25,  10.0),
            "qwen/qwen2.5-vl-72b-instruct":    (0.70,   0.70),
            "google/gemma-4-26b-a4b-it":       (0.10,   0.10),
            "google/gemma-4-31b-it":           (0.13,   0.38),
            "qwen/qwen3-vl-235b-a22b-instruct": (0.20,  0.88),
            "qwen/qwen3-vl-32b-instruct":      (0.10,   0.42),
        }
        pin, pout = pricing.get(model, (0.0, 0.0))
        cost_usd = (input_tokens * pin + output_tokens * pout) / 1_000_000
    cost_usd = float(cost_usd)
    return {
        "model":         model,
        "reply":         text,
        "latency_ms":    int(elapsed * 1000),
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "cost_usd":      round(cost_usd, 6),
        "provider":      "openrouter.ai",
        "raw":           resp,
    }


def encode_png(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Human-in-the-loop "backend"
# ---------------------------------------------------------------------------
#
# Lets an external party (an operator, a Claude Code chat session,
# a separate process) act as the VLM by writing JSON replies to
# files in a pending dir.  The harness behaves identically -- it
# still calls call_oracle on each turn and expects a JSON reply;
# the only difference is that the reply comes from a file, not an
# API.  Useful for:
#   * eyeballing one round of a game without spending API budget
#   * comparing a human/external-LLM strategy against the API VLM
#   * debugging the harness's prompt-construction in isolation
#
# Protocol (per call_oracle invocation, sequential within a session):
#   1. backend reads $VLM_HUMAN_PENDING_DIR (required); fails fast if unset
#   2. backend assigns call N (next available slot) and writes:
#        call_NNN_prompt.md   <- system + user prompt, plain text
#        call_NNN_image.png   <- the frame PNG (only if image_b64 set)
#   3. backend polls $VLM_HUMAN_PENDING_DIR/call_NNN_reply.json
#      every ~2s, up to VLM_HUMAN_TIMEOUT_S (default 7200s = 2h)
#   4. when the reply file appears non-empty, backend reads it,
#      moves it to call_NNN_reply.consumed.json, returns it
#
# The reply file must contain a single line that is the VLM's
# JSON reply (same shape the harness's parser expects -- tool_call
# or act).

def call_human_vlm(
    *,
    model:     str,
    system:    str,
    user:      str,
    image_b64: Optional[str] = None,
) -> dict:
    pending = os.environ.get("VLM_HUMAN_PENDING_DIR")
    if not pending:
        raise RuntimeError(
            "human-VLM backend selected but VLM_HUMAN_PENDING_DIR "
            "is not set.  The harness should export it before any "
            "call_oracle call."
        )
    pdir = Path(pending)
    pdir.mkdir(parents=True, exist_ok=True)
    # Sequence the call: count existing prompt files.
    existing = sorted(pdir.glob("call_*_prompt.md"))
    n = len(existing) + 1
    prompt_path = pdir / f"call_{n:03d}_prompt.md"
    image_path  = pdir / f"call_{n:03d}_image.png"
    reply_path  = pdir / f"call_{n:03d}_reply.json"
    # Atomic-ish status sentinel for the operator's CLI/UI to read.
    status_path = pdir / "STATUS.txt"
    prompt_text = (
        f"# Call #{n}\n\n"
        f"## SYSTEM PROMPT\n\n```\n{system}\n```\n\n"
        f"---\n\n"
        f"## USER PROMPT\n\n```\n{user}\n```\n\n"
        f"---\n\n"
        f"## Reply instructions\n\n"
        f"Write your JSON reply to:\n"
        f"  `{reply_path.name}`\n\n"
        f"Examples:\n"
        f"  tool_call: `{{\"action\":\"tool_call\","
        f"\"tool\":\"list_palettes\",\"args\":{{}}}}`\n"
        f"  act:       `{{\"action\":\"act\","
        f"\"action_id\":3,\"click_xy\":null,"
        f"\"reasoning\":\"...\",\"current_understanding\":\"...\","
        f"\"conclusions\":[\"...\"]}}`\n"
    )
    prompt_path.write_text(prompt_text, encoding="utf-8")
    if image_b64:
        try:
            image_path.write_bytes(base64.b64decode(image_b64))
        except Exception:
            pass
    has_image = " + image" if image_b64 else ""
    status_path.write_text(
        f"WAITING FOR REPLY on call_{n:03d}{has_image}\n"
        f"Wrote prompt to: {prompt_path}\n"
        f"Reply expected at: {reply_path}\n",
        encoding="utf-8",
    )
    # 30-minute default: long enough for a slow operator to read a
    # tool result and write a reply, short enough that a forgotten
    # session doesn't sit waiting for 2 hours.  Override with the
    # env var when long deliberation is genuinely expected.
    timeout_s = int(os.environ.get("VLM_HUMAN_TIMEOUT_S", "1800"))
    poll_s    = float(os.environ.get("VLM_HUMAN_POLL_S",   "2"))
    start = time.time()
    print(
        f"[call_human_vlm] waiting for reply at {reply_path} "
        f"(timeout {timeout_s}s, poll {poll_s}s)"
    )
    body = ""
    while time.time() - start < timeout_s:
        if reply_path.exists():
            try:
                body = reply_path.read_text(encoding="utf-8").strip()
            except Exception:
                body = ""
            if body:
                break
        time.sleep(poll_s)
    if not body:
        raise TimeoutError(
            f"human VLM did not write {reply_path} within "
            f"{timeout_s}s"
        )
    # Archive the consumed reply so the operator can see what
    # was returned, but a stale file from a prior call doesn't
    # accidentally get re-served.
    try:
        consumed = pdir / f"call_{n:03d}_reply.consumed.json"
        reply_path.rename(consumed)
    except Exception:
        pass
    status_path.write_text(
        f"RECEIVED reply for call_{n:03d} "
        f"({len(body)} chars after {int(time.time() - start)}s)\n",
        encoding="utf-8",
    )
    return {
        "model":         model,
        "reply":         body,
        "latency_ms":    int((time.time() - start) * 1000),
        "input_tokens":  0,
        "output_tokens": 0,
        "cost_usd":      0.0,
        "provider":      "human-in-the-loop",
        "raw":           None,
    }
