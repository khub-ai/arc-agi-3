"""Minimal OpenAI-compatible server backed by a local `transformers` VLM.

Stand-in for vLLM on the Kaggle image (which ships torch + transformers +
accelerate but NOT vLLM, and the scored run is offline so nothing can be
pip-installed). Loads a vision-language model in fp16 across the available GPUs
(`device_map="auto"`) and serves the two endpoints COS needs:

  GET  /v1/models            -> lists the served model id
  POST /v1/chat/completions  -> OpenAI chat, accepting text + image_url (base64
                                data URL) content parts -- exactly what
                                `vllm_backend` posts -- so COS's existing
                                `vllm/<host:port>/<name>` slug works unchanged.

Generation is serialized (one GPU, one decode at a time) behind a lock.

Usage: python serve_vlm.py --model <dir> --port 8000 --name qwen3-vl-8b
"""
from __future__ import annotations

import argparse
import base64
import io
import threading
import time

import torch
from PIL import Image
from flask import Flask, jsonify, request


def _load(model_dir: str):
    """Load the VLM, trying the modern generic class first, then fallbacks.

    Uses bf16 where the GPU supports it (Blackwell/Ampere+, e.g. RTX PRO 6000) --
    the dtype these models are trained in -- and falls back to fp16 on Turing (T4)."""
    import transformers
    dtype = (torch.bfloat16 if torch.cuda.is_available()
             and torch.cuda.is_bf16_supported() else torch.float16)
    print(f"[serve_vlm] dtype={dtype}", flush=True)
    last = None
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq",
                     "AutoModelForCausalLM"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            m = cls.from_pretrained(model_dir, dtype=dtype,
                                    device_map="auto", trust_remote_code=True)
            print(f"[serve_vlm] loaded via {cls_name}", flush=True)
            return m
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"[serve_vlm] {cls_name} failed: {type(e).__name__}: {e}", flush=True)
    raise last


def _data_url_to_image(url: str) -> Image.Image:
    b64 = url.split(",", 1)[1] if "," in url else url
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _to_conversation(messages):
    """OpenAI messages -> transformers chat conversation (image parts as PIL)."""
    conv = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            parts = [{"type": "text", "text": content}]
        else:
            parts = []
            for c in content or []:
                if c.get("type") == "text":
                    parts.append({"type": "text", "text": c.get("text", "")})
                elif c.get("type") == "image_url":
                    url = (c.get("image_url") or {}).get("url", "")
                    if url:
                        parts.append({"type": "image", "image": _data_url_to_image(url)})
        conv.append({"role": m.get("role", "user"), "content": parts})
    return conv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--name", default="local-vlm")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    from transformers import AutoProcessor
    print(f"[serve_vlm] loading {args.model} ...", flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = _load(args.model)
    model.eval()
    in_dev = getattr(model, "device", None) or "cuda:0"
    print(f"[serve_vlm] ready in {time.time() - t0:.0f}s (input device {in_dev})", flush=True)

    lock = threading.Lock()
    app = Flask(__name__)

    @app.get("/v1/models")
    def models():
        return jsonify({"object": "list", "data": [{"id": args.name, "object": "model"}]})

    @app.post("/v1/chat/completions")
    def chat():
        body = request.get_json(force=True)
        conv = _to_conversation(body.get("messages", []))
        max_new = int(body.get("max_tokens") or args.max_new_tokens)
        temp = float(body.get("temperature") or 0.0)
        with lock:
            inputs = processor.apply_chat_template(
                conv, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(in_dev)
            gen_kwargs = {"max_new_tokens": max_new}
            if temp > 0:
                gen_kwargs.update(do_sample=True, temperature=temp)
            else:
                gen_kwargs.update(do_sample=False)
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
            plen = inputs["input_ids"].shape[1]
            text = processor.batch_decode(out[:, plen:], skip_special_tokens=True)[0]
        return jsonify({
            "id": "chatcmpl-local", "object": "chat.completion", "model": args.name,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": int(plen),
                      "completion_tokens": int(out.shape[1] - plen),
                      "total_tokens": int(out.shape[1])},
        })

    print(f"[serve_vlm] serving '{args.name}' on 127.0.0.1:{args.port}", flush=True)
    app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
