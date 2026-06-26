"""Local vLLM (OpenAI-compatible) backend route for the offline submission.

`backends.call_oracle` has NO local-vLLM route: `google/gemma-*` slugs go to
OpenRouter/Together (cloud), which the offline Kaggle sandbox can't reach. For the
offline submission serving gemma-4-31b via **vLLM on localhost**, we install a thin
wrapper that intercepts `vllm/<host:port>/<served-model-name>` slugs and POSTs to
the local OpenAI-compatible endpoint, falling through to the original routing for
every other slug.

Submission-side only (monkeypatch from this package) - no edit to the shared
backends.py. cos_responder does `import backends; backends.call_oracle(...)`, so
patching the module attribute is picked up.

Usage (notebook, once, before the agent runs):
    os.environ["QWEN_MODEL_SLUG"] = "vllm/127.0.0.1:8000/gemma-4-31b-it"
    import vllm_backend; vllm_backend.install()
"""
from __future__ import annotations

import json
import re
import urllib.request

_SLUG = re.compile(r"vllm/([^/]+)/(.+)")


def _call_vllm(*, model, system, user, image_b64=None, max_tokens=2000,
               temperature=0.0, timeout_s=300, **_ignored):
    m = _SLUG.match(model)
    host, served = (m.group(1), m.group(2)) if m else ("127.0.0.1:8000", model)
    content = [{"type": "text", "text": user}]
    if image_b64:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
    body = json.dumps({
        "model": served,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": content}],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }).encode()
    req = urllib.request.Request(
        f"http://{host}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer local"})
    r = json.load(urllib.request.urlopen(req, timeout=timeout_s))
    txt = ((r.get("choices") or [{}])[0].get("message", {}) or {}).get("content", "") or ""
    usage = r.get("usage", {}) or {}
    return {"text": txt, "reply": txt, "provider": "vllm",
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0)}


def install():
    """Wrap backends.call_oracle so `vllm/...` slugs hit the local endpoint.
    Idempotent; returns True if installed/already-installed."""
    import backends
    if getattr(backends, "_vllm_installed", False):
        return True
    _orig = backends.call_oracle

    def call_oracle(*, model, **kw):
        if isinstance(model, str) and model.startswith("vllm/"):
            return _call_vllm(model=model, **kw)
        return _orig(model=model, **kw)

    backends.call_oracle = call_oracle
    backends._vllm_installed = True
    return True
