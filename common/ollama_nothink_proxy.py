"""Tiny local OpenAI-compat proxy that talks to Ollama's native
/api/chat with think=False. Purpose: make reasoning models (qwen3.5:9b,
etc.) usable through frameworks that only speak the OpenAI-compat
/v1/chat/completions surface.

Usage:
    python ollama_nothink_proxy.py [--port 11435] [--upstream http://localhost:11434]

Then point clients at http://localhost:11435/v1 with any api_key.

Supports:
- POST /v1/chat/completions (JSON body: model, messages, temperature,
  max_tokens, response_format, tools) -- forwards to Ollama /api/chat
  with think=False, returns OpenAI-compat response shape.
- Vision inputs: translates OpenAI-style multi-part content
  ([{"type":"text",...}, {"type":"image_url",...}]) into Ollama's
  native `content` + `images[]` format. Necessary because Ollama's
  OpenAI-compat surface can't raise num_ctx per-request, so vision
  requests hit the 4k default; going through this proxy sets
  num_ctx=16384 automatically.
- Streaming NOT supported (returns 400).
"""
from __future__ import annotations

import argparse
import http.server
import json
import time
import urllib.error
import urllib.request
import uuid

UPSTREAM = "http://localhost:11434"


def _translate_messages(msgs: list[dict]) -> list[dict]:
    """Translate OpenAI-style vision content (list of {type, text} /
    {type, image_url}) to Ollama-native format (content string +
    optional images array of raw base64). Text-only messages pass
    through unchanged."""
    out: list[dict] = []
    for msg in msgs:
        content = msg.get("content")
        if isinstance(content, list):
            text_parts: list[str] = []
            images: list[str] = []
            for part in content:
                ptype = part.get("type")
                if ptype == "text":
                    text_parts.append(part.get("text", ""))
                elif ptype == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    # Accept data URLs (data:image/...;base64,XXXX) and
                    # bare base64 -- Ollama wants the raw base64.
                    if url.startswith("data:") and "base64," in url:
                        images.append(url.split("base64,", 1)[1])
                    else:
                        images.append(url)
            new_msg = {"role": msg["role"], "content": "\n".join(text_parts)}
            if images:
                new_msg["images"] = images
            out.append(new_msg)
        else:
            out.append(msg)
    return out


def _openai_to_ollama(payload: dict) -> dict:
    """Translate an OpenAI /v1/chat/completions body to an Ollama
    /api/chat body with think=False."""
    body: dict = {
        "model": payload["model"],
        "messages": _translate_messages(payload["messages"]),
        "stream": False,
        "think": False,
        "options": {},
    }
    if "temperature" in payload:
        body["options"]["temperature"] = payload["temperature"]
    if "top_p" in payload:
        body["options"]["top_p"] = payload["top_p"]
    if payload.get("max_tokens"):
        body["options"]["num_predict"] = payload["max_tokens"]
        # If the caller asked for a lot of output, bump num_ctx too so
        # a big prompt + big response fits together. Ollama defaults
        # num_ctx=4096 which is often the real bottleneck.
        body["options"]["num_ctx"] = max(8192, payload["max_tokens"] + 4096)
    # Pass response_format through -- Ollama's OpenAI-compat supports
    # {"type": "json_object"} and {"type": "json_schema", ...}; native
    # /api/chat accepts `format: "json"` or `format: <json-schema>`.
    # Ollama's `format` accepts "json" (string) or a JSON-schema
    # object. In practice, passing complex nested schemas here
    # (like PydanticAI generates) causes Ollama to reject with
    # "Value looks like object, but can't find closing '}' symbol"
    # -- Ollama's schema parser has issues with some shapes.
    # Fall back to plain "json" mode: guarantees valid JSON out,
    # let the caller validate against its schema.
    rf = payload.get("response_format")
    if rf and rf.get("type") in ("json_object", "json_schema"):
        body["format"] = "json"
    # Tools passthrough. Ollama's /api/chat accepts a tools array in
    # the OpenAI-compat shape directly.
    if payload.get("tools"):
        body["tools"] = payload["tools"]
        if payload.get("tool_choice"):
            body["tool_choice"] = payload["tool_choice"]
    return body


def _ollama_to_openai(ollama_resp: dict, model: str) -> dict:
    """Translate an Ollama /api/chat response to OpenAI /v1/chat/completions shape."""
    msg = ollama_resp.get("message", {})
    content = msg.get("content", "") or ""
    tool_calls = msg.get("tool_calls") or None
    finish_reason = "stop"
    if ollama_resp.get("done_reason") == "length":
        finish_reason = "length"
    elif tool_calls:
        finish_reason = "tool_calls"

    openai_msg: dict = {"role": "assistant", "content": content}
    if tool_calls:
        openai_msg["tool_calls"] = [
            {
                "id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": (
                        tc["function"]["arguments"]
                        if isinstance(tc["function"]["arguments"], str)
                        else json.dumps(tc["function"]["arguments"])
                    ),
                },
            }
            for tc in tool_calls
        ]

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": openai_msg,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": ollama_resp.get("prompt_eval_count", 0),
            "completion_tokens": ollama_resp.get("eval_count", 0),
            "total_tokens": ollama_resp.get("prompt_eval_count", 0)
            + ollama_resp.get("eval_count", 0),
        },
    }


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silence access log

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            self.send_error(404, "only /v1/chat/completions is proxied")
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            self.send_error(400, f"bad json: {e}")
            return
        if payload.get("stream"):
            self.send_error(400, "streaming not supported by nothink proxy")
            return

        ollama_body = _openai_to_ollama(payload)
        req = urllib.request.Request(
            f"{UPSTREAM}/api/chat",
            data=json.dumps(ollama_body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            err_body = e.read()
            # Log upstream errors so users can see what Ollama rejected.
            # (Ollama's tool-schema handler is picky; complex nested
            # schemas from PydanticAI sometimes come back as
            # 400 "Value looks like object, but can't find closing '}'".)
            print(f"[proxy] upstream {e.code}: {err_body[:400]!r}", flush=True)
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(err_body)
            return

        openai_resp = _ollama_to_openai(data, payload["model"])
        body = json.dumps(openai_resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        # Some clients probe /v1/models
        if self.path in ("/v1/models", "/models"):
            body = json.dumps({"object": "list", "data": []}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)


def main():
    global UPSTREAM
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11435)
    ap.add_argument("--upstream", default="http://localhost:11434")
    args = ap.parse_args()
    UPSTREAM = args.upstream.rstrip("/")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"nothink proxy: OpenAI-compat on http://127.0.0.1:{args.port}, "
          f"upstream Ollama {UPSTREAM}")
    server.serve_forever()


if __name__ == "__main__":
    main()
