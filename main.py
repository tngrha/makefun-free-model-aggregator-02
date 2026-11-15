#!/usr/bin/env python3
"""free-model-aggregator-02: one server, two client wires, live free models.

A long-running program (Python 3.13, standard library only). It reads the
trial brief at $MAKEFUN_PROMPT_FILE, registers its port and catalog with the
env at $MAKEFUN_ENV_URL, serves the env's client as an OpenAI server
(POST /v1/chat/completions) and an Anthropic one (POST /v1/messages), in JSON
and in server-sent events, and exits 0 when GET $MAKEFUN_ENV_URL/work says
the workload is done.

Every completion comes from a real free OpenRouter model dialled through the
grader's gateway: POST $OPENAI_BASE_URL/chat/completions with the trial's
dummy key from OPENAI_API_KEY (Authorization: Bearer). Anthropic-shaped
requests are translated into Chat Completions bodies and the gateway's answer
translated back; the gateway's Messages route is never called for these ids.

Routing: for each request, the free models that meet its requirements are
ranked by the brief's rating (best first), then by how little they have been
used so far (calls plus a penalty per refusal), then by the brief's order.
A 429 or 5xx is retried once on the same model, then the request fails over
to the next candidate; a 401 means the entrant's key is missing or refused,
which no other model can fix, so the prompt is given up at once. Recovered
refusals are reported in the x-makefun-upstream-failures header. A model
that refuses twice in a row is skipped while any other candidate exists.
At most 48 model calls are made in a trial (the gateway caps at 60).

Everything is a deterministic function of the brief, the requests and the
gateway's answers: no wall clock, no unseeded randomness, so a replay
leaves the same state export.

`python3 main.py --list` prints the catalog document and exits 0.
"""

import json
import os
import sys
import threading
from http.client import HTTPConnection, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.1
UPSTREAM_TIMEOUT_S = 30
MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_TRIAL_CALLS = 48
MAX_ATTEMPTS_PER_PROMPT = 10
DEFAULT_MAX_TOKENS = 1024
MAX_OUTPUT_TOKENS = 8192
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")
GATEWAY_URL = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
GATEWAY_KEY = os.environ.get("OPENAI_API_KEY", "")

MODELS_SCHEMA = "free-model-aggregator-02/models/v1"
PROMPTS_SCHEMA = "free-model-aggregator-02/prompts/v1"

OPENROUTER_PUBLIC_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
PROVIDER_ID = "openrouter"


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(v):
    return isinstance(v, dict)


def string_of(v):
    return v if isinstance(v, str) else ""


def number_or_none(v):
    return v if isinstance(v, (int, float)) and v == v and abs(v) != float("inf") else None


def bool_or_none(v):
    return v if isinstance(v, bool) else None


def number_or_zero(v):
    return v if isinstance(v, (int, float)) else 0


# ---------------------------------------------------------------- brief ---

def _object_end(text, start):
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text):
    """Every JSON object in `text` that carries a string `schema`, in order."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
                if is_record(value) and isinstance(value.get("schema"), str):
                    out.append(value)
            except Exception:
                pass
        start = text.find("{", start + 1)
    return out


def _model_of(v):
    if not is_record(v):
        return None
    mid = string_of(v.get("id"))
    if mid == "":
        return None
    rating = v.get("rating_milli")
    return {
        "id": mid,
        "provider": PROVIDER_ID,
        "display_name": string_of(v.get("display_name")) or mid,
        "rating_milli": rating if isinstance(rating, (int, float)) else 0,
        "context_length": number_or_none(v.get("context_length")),
        "tools": v.get("tools") is True,
        "vision": v.get("vision") is True,
        "json": v.get("json") is True,
        "free": v.get("free") is not False,
    }


def _prompt_of(v):
    if not is_record(v):
        return None
    pid = string_of(v.get("id"))
    if pid == "":
        return None
    req = v.get("requires") if is_record(v.get("requires")) else {}
    out_req = {
        "tools": req.get("tools") is True,
        "vision": req.get("vision") is True,
        "json": req.get("json") is True,
    }
    mc = req.get("min_context")
    if isinstance(mc, (int, float)):
        out_req["min_context"] = mc
    return {
        "id": pid,
        "api": "anthropic" if v.get("api") == "anthropic" else "openai",
        "stream": v.get("stream") is True,
        "weight": 2 if v.get("weight") == 2 else 1,
        "requires": out_req,
    }


def read_brief(path):
    """Returns (models, prompts, error)."""
    if path == "":
        return [], [], "MAKEFUN_PROMPT_FILE is not set"
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        return [], [], "cannot read %s: %s" % (path, e)
    docs = documents_in(text)
    models = []
    for d in docs:
        if d.get("schema") == MODELS_SCHEMA and isinstance(d.get("models"), list):
            for v in d["models"]:
                m = _model_of(v)
                if m is not None:
                    models.append(m)
            break
    if not models:
        return [], [], "the brief holds no %s document" % MODELS_SCHEMA
    prompts = []
    for d in docs:
        if d.get("schema") == PROMPTS_SCHEMA and isinstance(d.get("prompts"), list):
            for v in d["prompts"]:
                p = _prompt_of(v)
                if p is not None:
                    prompts.append(p)
            break
    return models, prompts, None


# -------------------------------------------------------------- catalog ---

def build_catalog(models):
    """The catalog document: one provider, exactly the brief's free models."""
    seen = set()
    rows = []
    for model in models:
        if not model["free"] or model["id"] in seen:
            continue
        seen.add(model["id"])
        rows.append({"id": model["id"], "free": True,
                     "context_length": model["context_length"]})
    if not rows:
        return {"providers": []}
    return {"providers": [{"id": PROVIDER_ID,
                           "base_url": OPENROUTER_PUBLIC_BASE,
                           "key_env": OPENROUTER_KEY_ENV,
                           "wire": "openai",
                           "models": rows}]}


def catalog_bytes(catalog):
    return json.dumps(catalog, separators=(",", ":")).encode("utf-8")


# ------------------------------------------------------- wire translation ---

def text_of(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for b in content:
        if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str):
            parts.append(b["text"])
    return "\n".join(parts)


def part_of(block):
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    if block.get("type") == "image" and is_record(block.get("source")):
        source = block["source"]
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
            return {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != "" else json.dumps(content)
    return "" if content is None else json.dumps(content)


def arguments_of(value):
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
            if is_record(parsed):
                return parsed
        except Exception:
            pass
    return {}


def stop_reason_of_openai(finish):
    """OpenAI finish_reason -> Anthropic stop_reason."""
    if finish == "length":
        return "max_tokens"
    if finish in ("tool_calls", "function_call"):
        return "tool_use"
    if finish == "content_filter":
        return "refusal"
    return "end_turn"


def finish_reason_of_anthropic(stop):
    """Anthropic stop_reason -> OpenAI finish_reason."""
    if stop == "max_tokens":
        return "length"
    if stop == "tool_use":
        return "tool_calls"
    if stop == "refusal":
        return "content_filter"
    return "stop"


def anthropic_to_openai(body, model, json_mode=False):
    """An Anthropic Messages request as an OpenAI Chat Completions request."""
    messages = []
    system = text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    source = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls = []
            for block in content:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append({
                    "id": block.get("id") if isinstance(block.get("id"), str) else "",
                    "type": "function",
                    "function": {
                        "name": block.get("name") if isinstance(block.get("name"), str) else "",
                        "arguments": json.dumps(block.get("input") if is_record(block.get("input")) else {}),
                    },
                })
            text = text_of(content)
            msg = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                    "content": tool_result_text(block.get("content")),
                })
                continue
            part = part_of(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
        out["stop"] = body["stop_sequences"]
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            tools.append({
                "type": "function",
                "function": {
                    "name": raw["name"],
                    "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                    "parameters": raw.get("input_schema") if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
                },
            })
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    if json_mode:
        out["response_format"] = {"type": "json_object"}
    return out


def openai_to_anthropic(body, model):
    """A Chat Completions answer as an Anthropic Messages object."""
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    content = []
    text = message.get("content") if isinstance(message.get("content"), str) else text_of(message.get("content"))
    if text != "":
        content.append({"type": "text", "text": text})
    calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    for call in calls:
        if not is_record(call) or not is_record(call.get("function")):
            continue
        content.append({
            "type": "tool_use",
            "id": call.get("id") if isinstance(call.get("id"), str) else "toolu_unknown",
            "name": call["function"].get("name") if isinstance(call["function"].get("name"), str) else "",
            "input": arguments_of(call["function"].get("arguments")),
        })
    if not content:
        content.append({"type": "text", "text": ""})
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    return {
        "id": "msg_" + (body.get("id") if isinstance(body.get("id"), str) else "aggregate"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason_of_openai(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {"input_tokens": number_or_zero(usage.get("prompt_tokens")),
                  "output_tokens": number_or_zero(usage.get("completion_tokens"))},
    }


def anthropic_message_to_openai(body, model, request_seq):
    """An Anthropic Messages object as a Chat Completions object."""
    content = body.get("content") if isinstance(body.get("content"), list) else []
    texts = []
    tool_calls = []
    for block in content:
        if not is_record(block):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id") if isinstance(block.get("id"), str) else "toolu_unknown",
                "type": "function",
                "function": {
                    "name": block.get("name") if isinstance(block.get("name"), str) else "",
                    "arguments": json.dumps(block.get("input") if is_record(block.get("input")) else {}),
                },
            })
    text = "".join(texts)
    message = {"role": "assistant", "content": text if text != "" else None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    prompt_tokens = number_or_zero(usage.get("input_tokens"))
    completion_tokens = number_or_zero(usage.get("completion_tokens"))
    return {
        "id": "chatcmpl-" + (body.get("id") if isinstance(body.get("id"), str) else "agg-%d" % request_seq),
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": finish_reason_of_anthropic(body.get("stop_reason"))}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


# ------------------------------------------------------------------ SSE ---

def sse_event(event, data):
    prefix = "" if event is None else "event: %s\n" % event
    return "%sdata: %s\n\n" % (prefix, json.dumps(data, separators=(",", ":")))


def parse_sse(text):
    """SSE text -> list of data payload strings ('[DONE]' included as-is)."""
    out = []
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        frame = frame.strip("\n")
        if frame == "":
            continue
        for line in frame.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                out.append(line[5:].lstrip())
    return out


def openai_chunk(model, request_seq, delta=None, finish_reason=None, usage=None, chunk_id=None):
    chunk = {
        "id": chunk_id or ("chatcmpl-agg-%d" % request_seq),
        "object": "chat.completion.chunk",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def openai_chunks_to_anthropic_events(chunks, model, request_seq):
    """OpenAI chat chunk dicts -> Anthropic SSE event strings."""
    events = []
    started = False
    finished = False
    pending_stop = None
    text_index = -1
    next_index = 0
    tool_indexes = {}
    input_tokens = 0
    output_tokens = 0
    for _chunk in chunks:
        _usage = _chunk.get("usage") if is_record(_chunk) else None
        if is_record(_usage):
            input_tokens = number_or_zero(_usage.get("prompt_tokens"))
            output_tokens = number_or_zero(_usage.get("completion_tokens"))

    def start_events():
        nonlocal started
        if started:
            return []
        started = True
        return [sse_event("message_start", {
            "type": "message_start",
            "message": {"id": "msg_stream", "type": "message", "role": "assistant", "model": model,
                        "content": [], "stop_reason": None, "stop_sequence": None,
                        "usage": {"input_tokens": input_tokens, "output_tokens": 0}}})]

    def close_events():
        nonlocal text_index
        out = []
        if text_index != -1:
            out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": text_index}))
            text_index = -1
        for index in sorted(tool_indexes.values()):
            out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": index}))
        tool_indexes.clear()
        return out

    def finish_events(stop_reason):
        nonlocal finished
        if finished:
            return []
        finished = True
        return close_events() + [
            sse_event("message_delta", {"type": "message_delta",
                                        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                                        "usage": {"output_tokens": output_tokens}}),
            sse_event("message_stop", {"type": "message_stop"}),
        ]

    for chunk in chunks:
        if not is_record(chunk):
            continue
        usage = chunk.get("usage") if is_record(chunk.get("usage")) else {}
        if usage:
            input_tokens = number_or_zero(usage.get("prompt_tokens"))
            output_tokens = number_or_zero(usage.get("completion_tokens"))
        choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
        choice = choices[0] if choices and is_record(choices[0]) else None
        if choice is None:
            continue
        delta = choice.get("delta") if is_record(choice.get("delta")) else {}
        events.extend(start_events())
        if isinstance(delta.get("content"), str) and delta["content"] != "":
            if text_index == -1:
                text_index = next_index
                next_index += 1
                events.append(sse_event("content_block_start", {
                    "type": "content_block_start", "index": text_index,
                    "content_block": {"type": "text", "text": ""}}))
            events.append(sse_event("content_block_delta", {
                "type": "content_block_delta", "index": text_index,
                "delta": {"type": "text_delta", "text": delta["content"]}}))
        calls = delta.get("tool_calls") if isinstance(delta.get("tool_calls"), list) else []
        for call in calls:
            if not is_record(call):
                continue
            upstream_index = call.get("index") if isinstance(call.get("index"), int) else len(tool_indexes)
            index = tool_indexes.get(upstream_index)
            if index is None:
                if text_index != -1:
                    events.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": text_index}))
                    text_index = -1
                index = next_index
                next_index += 1
                tool_indexes[upstream_index] = index
                fn = call.get("function") if is_record(call.get("function")) else {}
                events.append(sse_event("content_block_start", {
                    "type": "content_block_start", "index": index,
                    "content_block": {"type": "tool_use",
                                      "id": call.get("id") if isinstance(call.get("id"), str) and call.get("id") != "" else "toolu_%d" % index,
                                      "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                                      "input": {}}}))
            fn = call.get("function") if is_record(call.get("function")) else {}
            if isinstance(fn.get("arguments"), str) and fn["arguments"] != "":
                events.append(sse_event("content_block_delta", {
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]}}))
        if choice.get("finish_reason") is not None:
            pending_stop = stop_reason_of_openai(choice["finish_reason"])
    if not finished:
        events.extend(start_events())
        events.extend(finish_events(pending_stop or "end_turn"))
    return events


# ------------------------------------------------------------ server state ---

MODELS, PROMPTS, BRIEF_ERROR = read_brief(PROMPT_FILE)
CATALOG = build_catalog(MODELS)
FREE_MODELS = [m for m in MODELS if m["free"]]

calls = {}                # model id -> gateway calls made so far (this trial)
refusals = {}             # model id -> refusals so far
consecutive_refusals = {}  # model id -> refusals in a row, reset by a success
exhausted = set()         # models that refused twice in a row
total_calls = 0
manifest_index = 0
request_seq = 0
manifest_lock = threading.Lock()


def capabilities_of(model):
    return {
        "context_length": model["context_length"],
        "tools": model["tools"],
        "vision": model["vision"],
        "json": model["json"],
    }


def meets_requirements(requires, caps):
    if requires.get("tools") is True and caps["tools"] is not True:
        return False
    if requires.get("vision") is True and caps["vision"] is not True:
        return False
    if requires.get("json") is True and caps["json"] is not True:
        return False
    mc = requires.get("min_context")
    if mc is not None and (caps["context_length"] is None or caps["context_length"] < mc):
        return False
    return True


def candidates_for(requires):
    """Free models meeting the requirements, ranked best first: by rating,
    then by how little they have been used (calls plus three per refusal),
    then by the brief's order, then by model id."""
    out = []
    for order, model in enumerate(FREE_MODELS):
        if not meets_requirements(requires, capabilities_of(model)):
            continue
        out.append({
            "model": model,
            "rating": model["rating_milli"],
            "order": (order, model["id"]),
        })
    out.sort(key=lambda c: (-c["rating"],
                            calls.get(c["model"]["id"], 0) + 3 * refusals.get(c["model"]["id"], 0),
                            c["order"][0], c["order"][1]))
    fresh = [c for c in out if c["model"]["id"] not in exhausted]
    return fresh if fresh else out


def requirements_from_body(body):
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    vision = False
    for message in messages:
        if not is_record(message) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if is_record(block) and block.get("type") in ("image", "image_url"):
                vision = True
    fmt = body.get("response_format") if is_record(body.get("response_format")) else {}
    return {
        "tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
        "vision": vision,
        "json": fmt.get("type") in ("json_object", "json_schema"),
    }


def requirements_for(body):
    """The requirements of the request now arriving: the manifest's row by
    arrival order, else the body's own."""
    global manifest_index
    with manifest_lock:
        index = manifest_index
        manifest_index += 1
    entry = PROMPTS[index] if index < len(PROMPTS) else None
    if entry is not None:
        return entry["requires"]
    return requirements_from_body(body)


# ---------------------------------------------------------------- gateway ---

def http_request(method, url, headers=None, body=None, timeout=5):
    """One HTTP request. Returns (status, headers_dict, body_bytes). Raises on
    connection failure. Does not follow redirects."""
    parts = urlparse(url)
    conn_cls = HTTPSConnection if parts.scheme == "https" else HTTPConnection
    host = parts.hostname
    port = parts.port or (443 if parts.scheme == "https" else 80)
    conn = conn_cls(host, port, timeout=timeout)
    try:
        path = parts.path or "/"
        if parts.query:
            path += "?" + parts.query
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, dict(resp.getheaders()), data
    finally:
        conn.close()


def gateway_url():
    """The gateway's Chat Completions route: $OPENAI_BASE_URL is the
    OpenAI-compatible root, ending in /v1 on the trial network."""
    if GATEWAY_URL == "":
        return ""
    if GATEWAY_URL.endswith("/v1"):
        return GATEWAY_URL + "/chat/completions"
    return GATEWAY_URL + "/v1/chat/completions"


def ensure_output_cap(body):
    """The gateway refuses a call with no output cap or one over 8192."""
    if isinstance(body.get("max_tokens"), (int, float)):
        body["max_tokens"] = max(1, min(int(body["max_tokens"]), MAX_OUTPUT_TOKENS))
    elif isinstance(body.get("max_completion_tokens"), (int, float)):
        body["max_completion_tokens"] = max(1, min(int(body["max_completion_tokens"]), MAX_OUTPUT_TOKENS))
    else:
        body["max_tokens"] = DEFAULT_MAX_TOKENS
    return body


def build_upstream_body(client_wire, model_id, body, stream, json_mode):
    """The Chat Completions body for the dialled model, on the gateway."""
    if client_wire == "openai":
        out = dict(body)
        out["model"] = model_id
    else:
        out = anthropic_to_openai(body, model_id, json_mode=json_mode)
    ensure_output_cap(out)
    if stream:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def dial_gateway(model_id, body):
    """POST one request to the gateway. Returns (status, body_bytes)."""
    url = gateway_url()
    if url == "":
        raise RuntimeError("OPENAI_BASE_URL is not set")
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
    headers = {"content-type": "application/json",
               "authorization": "Bearer " + GATEWAY_KEY}
    if body.get("stream") is True:
        headers["accept"] = "text/event-stream"
    status, _hdrs, data = http_request("POST", url, headers=headers, body=payload,
                                       timeout=UPSTREAM_TIMEOUT_S)
    return status, data


def error_code_of(data):
    """The gateway error's code (or type) from a refusal body, else ''."""
    try:
        value = json.loads(data.decode("utf-8", "replace"))
    except Exception:
        return ""
    if not is_record(value):
        return ""
    err = value.get("error")
    if is_record(err):
        if isinstance(err.get("code"), str) and err["code"] != "":
            return err["code"][:64]
        if isinstance(err.get("type"), str) and err["type"] != "":
            return err["type"][:64]
    return ""


# ------------------------------------------------------------- rendering ---

def answer_text_of_openai(body):
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    if isinstance(message.get("content"), str):
        return message["content"]
    return text_of(message.get("content"))


def ensure_usage_json(body):
    """A Chat Completions answer whose usage the wire check accepts: a whole
    input count of at least 0 and an output count of at least 1. The
    provider's own usage is kept whenever it is present and complete; only
    a missing one is filled in from the answer's length."""
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    prompt_tokens = number_or_none(usage.get("prompt_tokens"))
    completion_tokens = number_or_none(usage.get("completion_tokens"))
    if prompt_tokens is None or completion_tokens is None or completion_tokens < 1:
        text = answer_text_of_openai(body)
        if prompt_tokens is None:
            prompt_tokens = 0
        if completion_tokens is None or completion_tokens < 1:
            completion_tokens = max(1, len(text) // 4)
    body["usage"] = {"prompt_tokens": int(prompt_tokens),
                     "completion_tokens": int(completion_tokens),
                     "total_tokens": int(prompt_tokens) + int(completion_tokens)}
    return body


def ensure_finish_reason(body):
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    if choices and is_record(choices[0]):
        if not isinstance(choices[0].get("finish_reason"), str):
            choices[0]["finish_reason"] = "stop"
    return body


def normalize_json_answer(data, model_id):
    """A gateway JSON answer as a Chat Completions object naming the model
    this entry dialled."""
    try:
        parsed = json.loads(data.decode("utf-8"))
    except Exception:
        parsed = None
    if not is_record(parsed):
        parsed = {}
    parsed["model"] = model_id
    if not isinstance(parsed.get("id"), str):
        parsed["id"] = "chatcmpl-agg"
    if not isinstance(parsed.get("object"), str):
        parsed["object"] = "chat.completion"
    if not isinstance(parsed.get("created"), (int, float)):
        parsed["created"] = FIXED_CREATED
    choices = parsed.get("choices") if isinstance(parsed.get("choices"), list) else []
    if not choices or not is_record(choices[0]):
        parsed["choices"] = [{"index": 0,
                              "message": {"role": "assistant", "content": ""},
                              "finish_reason": "stop"}]
    else:
        if not is_record(choices[0].get("message")):
            choices[0]["message"] = {"role": "assistant", "content": ""}
        if choices[0]["message"].get("role") != "assistant":
            choices[0]["message"]["role"] = "assistant"
    ensure_finish_reason(parsed)
    ensure_usage_json(parsed)
    return parsed


def normalize_stream_answer(data, model_id, request_seq):
    """A gateway SSE body as OpenAI chat chunk dicts, every chunk naming the
    model this entry dialled, with a final usage chunk when the provider
    sent none."""
    frames = [p for p in parse_sse(data.decode("utf-8", "replace")) if p != "[DONE]"]
    chunks = []
    for p in frames:
        try:
            chunk = json.loads(p)
        except Exception:
            continue
        if is_record(chunk):
            chunk["model"] = model_id
            chunks.append(chunk)
    has_usage = any(is_record(c.get("usage")) for c in chunks)
    has_finish = any(isinstance(c.get("choices"), list) and c["choices"]
                     and is_record(c["choices"][0])
                     and isinstance(c["choices"][0].get("finish_reason"), str)
                     for c in chunks)
    text = ""
    for c in chunks:
        choices = c.get("choices") if isinstance(c.get("choices"), list) else []
        if choices and is_record(choices[0]):
            delta = choices[0].get("delta") if is_record(choices[0].get("delta")) else {}
            if isinstance(delta.get("content"), str):
                text += delta["content"]
    if not chunks:
        chunks.append(openai_chunk(model_id, request_seq, delta={"role": "assistant"}))
    if not has_finish:
        chunks.append(openai_chunk(model_id, request_seq, delta={}, finish_reason="stop"))
    if not has_usage:
        chunks.append(openai_chunk(model_id, request_seq, delta={}, finish_reason=None,
                                   usage={"prompt_tokens": 0,
                                          "completion_tokens": max(1, len(text) // 4),
                                          "total_tokens": max(1, len(text) // 4)}))
    return chunks


def render_to_client(client_wire, model_id, stream, normalized, request_seq):
    """A normalized Chat Completions answer in the wire the client used."""
    if not stream:
        if client_wire == "openai":
            return 200, "application/json", json.dumps(normalized, separators=(",", ":")).encode("utf-8")
        out = openai_to_anthropic(normalized, model_id)
        return 200, "application/json", json.dumps(out, separators=(",", ":")).encode("utf-8")
    chunks = normalized
    if client_wire == "openai":
        parts = []
        for chunk in chunks:
            parts.append("data: %s\n\n" % json.dumps(chunk, separators=(",", ":")))
        parts.append("data: [DONE]\n\n")
        return 200, "text/event-stream", "".join(parts).encode("utf-8")
    events = openai_chunks_to_anthropic_events(chunks, model_id, request_seq)
    return 200, "text/event-stream", "".join(events).encode("utf-8")


def serve(client_wire, body):
    """Answer one client request. Returns (status, content_type, body_bytes,
    failures) where failures is the trail of recovered refusals for the
    x-makefun-upstream-failures header (possibly empty)."""
    global request_seq, total_calls
    seq = request_seq
    request_seq += 1
    requires = requirements_for(body)
    stream = body.get("stream") is True
    json_mode = requires.get("json") is True
    failures = []
    attempts = 0
    for cand in candidates_for(requires):
        model_id = cand["model"]["id"]
        upstream_body = build_upstream_body(client_wire, model_id, body, stream, json_mode)
        tried_same = 0
        while tried_same < 2:
            if total_calls >= MAX_TRIAL_CALLS or attempts >= MAX_ATTEMPTS_PER_PROMPT:
                break
            attempts += 1
            total_calls += 1
            try:
                status, data = dial_gateway(model_id, upstream_body)
            except Exception as e:
                log("call to %s failed: %s" % (model_id, e))
                failures.append({"status": 502, "code": "upstream_unreachable", "model": model_id})
                break
            calls[model_id] = calls.get(model_id, 0) + 1
            if 200 <= status < 300:
                consecutive_refusals[model_id] = 0
                if stream:
                    normalized = normalize_stream_answer(data, model_id, seq)
                else:
                    normalized = normalize_json_answer(data, model_id)
                status_out, ctype, payload = render_to_client(
                    client_wire, model_id, stream, normalized, seq)
                return status_out, ctype, payload, failures
            code = error_code_of(data)
            failures.append({"status": status, "code": code, "model": model_id})
            refusals[model_id] = refusals.get(model_id, 0) + 1
            consecutive_refusals[model_id] = consecutive_refusals.get(model_id, 0) + 1
            if status >= 500 and consecutive_refusals[model_id] >= 2:
                # A 5xx twice in a row looks like an outage: stop spending
                # calls on this model while any other candidate exists. A
                # 429 is only a rate limit, which clears quickly, so it
                # never exhausts a model on its own.
                exhausted.add(model_id)
                log("model %s refused twice in a row, skipping it from now on" % model_id)
            log("model %s refused with HTTP %d (%s)" % (model_id, status, code))
            if status == 401:
                # No key (or a refused key): no other model can fix that.
                break
            if status in (429, 500, 502, 503, 504) and tried_same == 0:
                tried_same += 1
                continue
            break
        if failures and failures[-1]["status"] == 401:
            break
        if total_calls >= MAX_TRIAL_CALLS or attempts >= MAX_ATTEMPTS_PER_PROMPT:
            break
    if client_wire == "anthropic":
        err = {"type": "error", "error": {"type": "overloaded_error",
                                           "message": "no free model on the directory meets this request"}}
    else:
        err = {"error": {"message": "no free model on the directory meets this request",
                         "type": "no_free_model", "code": None}}
    return 503, "application/json", json.dumps(err, separators=(",", ":")).encode("utf-8"), failures


# ---------------------------------------------------------------- handler ---

def send_json(handler, status, body, failures=None):
    text = json.dumps(body, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("content-type", "application/json")
    handler.send_header("content-length", str(len(text)))
    if failures:
        send_failures_header(handler, failures)
    handler.end_headers()
    handler.wfile.write(text)


def send_failures_header(handler, failures):
    rows = failures[:8]
    try:
        value = json.dumps(rows, separators=(",", ":"))
    except Exception:
        return
    if len(value) > 8192:
        return
    handler.send_header("x-makefun-upstream-failures", value)


def send_error(handler, wire, status, err_type, message, failures=None):
    if wire == "anthropic":
        send_json(handler, status, {"type": "error", "error": {"type": err_type, "message": message}}, failures)
    else:
        send_json(handler, status, {"error": {"message": message, "type": err_type, "code": None}}, failures)


def openai_model_list(models):
    return {"object": "list",
            "data": [{"id": m["id"], "object": "model", "created": FIXED_CREATED,
                      "owned_by": m["provider"]} for m in models]}


def anthropic_model_list(models):
    return {"data": [{"type": "model", "id": m["id"], "display_name": m["display_name"],
                      "created_at": FIXED_ANTHROPIC_CREATED_AT} for m in models],
            "has_more": False}


def served_models():
    """The free models the brief lists, as GET /v1/models names them."""
    return [{"id": m["id"], "provider": m["provider"], "display_name": m["display_name"]}
            for m in FREE_MODELS]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if length <= 0 or length > MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        return value if is_record(value) else None

    def _send(self, status, content_type, data, failures=None):
        try:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(data)))
            if failures:
                send_failures_header(self, failures)
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/v1/models":
            models = served_models()
            if isinstance(self.headers.get("x-api-key"), str):
                send_json(self, 200, anthropic_model_list(models))
            else:
                send_json(self, 200, openai_model_list(models))
            return
        send_error(self, "openai", 404, "not_found", "no route GET %s" % self.path)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/v1/chat/completions":
            wire = "openai"
        elif path == "/v1/messages":
            wire = "anthropic"
        else:
            send_error(self, "anthropic" if path == "/v1/messages" else "openai",
                       404, "not_found", "no route POST %s" % path)
            return
        body = self._read_json()
        if body is None:
            send_error(self, wire, 400, "invalid_request_error",
                       "the request body is not one JSON object")
            return
        try:
            status, content_type, data, failures = serve(wire, body)
        except Exception as e:
            log("serve failed: %s" % e)
            send_error(self, wire, 500, "internal_error", "the server failed to answer")
            return
        self._send(status, content_type, data, failures)


# ------------------------------------------------------------------- main ---

def register(port):
    res_status, _hdrs, data = http_request(
        "POST", ENV_URL + "/register",
        headers={"content-type": "application/json"},
        body=json.dumps({"protocol_version": PROTOCOL_VERSION, "port": port,
                         "catalog": CATALOG}, separators=(",", ":")).encode("utf-8"),
        timeout=5)
    if res_status < 200 or res_status >= 300:
        raise RuntimeError("POST %s/register answered HTTP %d: %s"
                           % (ENV_URL, res_status, data[:500].decode("utf-8", "replace")))
    try:
        reply = json.loads(data.decode("utf-8"))
    except Exception:
        reply = {}
    if is_record(reply) and isinstance(reply.get("requests"), (int, float)):
        log("registered: the workload holds %d requests" % reply["requests"])


def poll_work(server):
    while True:
        threading.Event().wait(POLL_INTERVAL_S)
        try:
            status, _hdrs, data = http_request("GET", ENV_URL + "/work", timeout=2)
            if status == 200:
                work = json.loads(data.decode("utf-8"))
                if is_record(work) and work.get("done") is True:
                    server.shutdown()
                    return
        except Exception:
            pass


def main():
    if BRIEF_ERROR is not None:
        log(BRIEF_ERROR)
    if "--list" in sys.argv:
        sys.stdout.write(json.dumps(CATALOG, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        return
    if ENV_URL == "":
        log("MAKEFUN_ENV_URL is not set")
        sys.exit(1)
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    port = server.server_address[1]
    if port == 8080:
        log("the OS picked 8080, which the env owns on one layout: restart the trial")
        sys.exit(1)
    log("listening on 0.0.0.0:%d with %d free models"
        % (port, len(served_models())))
    register(port)
    threading.Thread(target=poll_work, args=(server,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(str(e))
        sys.exit(1)
