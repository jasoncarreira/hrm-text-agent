"""Prompt format + PrefixLM example construction for HRM-Text tool-use SFT.

HRM-Text is a PrefixLM with a *conditioning* scheme: prompts are
`<|im_start|>{condition}{task}<|im_end|>`. For structured output (tool calls) the
model's docs recommend the `direct` condition (token `<|object_ref_start|>`), so we
prepend it. The prompt span is bidirectional (token_type_ids=1); the completion is
causal (token_type_ids=0). Each conversation is split at every assistant turn into a
(prefix -> target) example.

Conversation schema (one JSON per line):
    {"tools": [<schema dict>, ...] | [],
     "turns": [{"role":"user","content":...},
               {"role":"calls","calls":[{"name","arguments"}],"observations":[...]},
               {"role":"final","content":...}, ...]}
Empty "tools" => a plain instruction example (no tools block) — teaches answering.
"""
from __future__ import annotations

import json
from typing import Any

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
BOX_END = "<|box_end|>"            # eos
TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
DIRECT = "<|object_ref_start|>"    # 'direct' condition: structured-output / no-CoT (HRM-Text docs)

ASSISTANT_ROLES = ("calls", "final")

SYSTEM = (
    "You are a function-calling assistant. You are given tool signatures inside "
    "<tools></tools>. To call tools, emit one or more "
    f"{TOOL_OPEN}{{\"name\": <tool>, \"arguments\": <args>}}{TOOL_CLOSE} blocks. "
    "After receiving tool results, either call more tools or give the user a final "
    "answer. If no tool is needed, answer directly."
)


def _tools_block(tool_schemas: list[dict]) -> str:
    lines = [json.dumps(s, separators=(",", ":")) for s in tool_schemas]
    return "<tools>\n" + "\n".join(lines) + "\n</tools>"


def serialize_call(name: str, arguments: dict) -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, separators=(",", ":"))
    return f"{TOOL_OPEN}{payload}{TOOL_CLOSE}"


def serialize_calls(calls: list[dict]) -> str:
    return "\n".join(serialize_call(c["name"], c.get("arguments", {})) for c in calls)


def serialize_assistant(turn: dict) -> str:
    if turn["role"] == "calls":
        return serialize_calls(turn["calls"])
    return turn["content"]


def _render_context(turns: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for t in turns:
        role = t["role"]
        if role == "user":
            lines.append(f"User: {t['content']}")
        elif role == "calls":
            lines.append(serialize_calls(t["calls"]))
            obs = t.get("observations") or []
            if obs:
                lines.append("Results:")
                lines.extend(str(o) for o in obs)
        elif role == "final":
            lines.append(f"Assistant: {t['content']}")
    return lines


def build_prefix(tool_schemas: list[dict], turns: list[dict[str, Any]]) -> str:
    """Render the dialogue so far as one bidirectional prefix block ending in <|im_end|>.

    Opens with the `direct` condition token. Omits the tools block when no tools are
    provided (plain instruction examples).
    """
    parts = [f"{IM_START}{DIRECT}{SYSTEM}", ""]
    if tool_schemas:
        parts += ["Available tools:", _tools_block(tool_schemas), ""]
    parts.extend(_render_context(turns))
    return "\n".join(parts) + IM_END


def expand_conversation(convo: dict) -> list[tuple[str, str]]:
    """Emit (prefix, target) examples for a conversation.

    A "raw" convo ({"raw_prompt", "raw_target"}) is passed through verbatim — it bypasses
    build_prefix, for format-discipline data in the *academic* envelope (single-letter MCQ,
    \\boxed{} math), which isn't tool-shaped. raw_prompt must already end in <|im_end|> and
    raw_target in <|box_end|>.
    """
    if "raw_prompt" in convo:
        return [(convo["raw_prompt"], convo["raw_target"])]
    examples: list[tuple[str, str]] = []
    tools = convo.get("tools", [])
    context: list[dict[str, Any]] = []
    for turn in convo["turns"]:
        if turn["role"] in ASSISTANT_ROLES:
            examples.append((build_prefix(tools, context), serialize_assistant(turn) + BOX_END))
        context.append(turn)
    return examples


def encode_example(tokenizer, prefix: str, target: str, max_len: int) -> dict | None:
    """Tokenize (prefix, target) with loss masking. Drops examples > max_len.

    token_type_ids: 1 over prefix (bidirectional), 0 over target (causal).
    labels: -100 over prefix, real ids over target (incl. eos, so it learns to stop).
    """
    pre = tokenizer.encode(prefix, add_special_tokens=False)
    tgt = tokenizer.encode(target, add_special_tokens=False)
    if len(pre) + len(tgt) > max_len:
        return None
    input_ids = pre + tgt
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "token_type_ids": [1] * len(pre) + [0] * len(tgt),
        "labels": [-100] * len(pre) + tgt,
    }
