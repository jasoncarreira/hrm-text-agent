"""Toy tool registry for HRM-Text tool-use fine-tuning.

Each tool has a JSON schema (rendered into the prompt so the model knows the
interface) and a Python implementation (used by the inference agent loop to
actually execute a predicted call). The dataset uses the same schemas, so the
training format and the runtime format stay in sync.
"""
from __future__ import annotations

import ast
import datetime as _dt
import operator as _op


# ---- implementations -------------------------------------------------------

# Deterministic canned data so the agent loop is reproducible offline.
_WEATHER = {
    "paris": {"temp_c": 18, "conditions": "partly cloudy"},
    "london": {"temp_c": 14, "conditions": "light rain"},
    "tokyo": {"temp_c": 23, "conditions": "clear"},
    "new york": {"temp_c": 21, "conditions": "sunny"},
    "san francisco": {"temp_c": 16, "conditions": "foggy"},
}

_SEARCH = {
    "hierarchical reasoning model": "HRM is a recurrent architecture with two "
        "coupled modules (slow high-level planner, fast low-level worker) that "
        "iterate over shared embeddings to gain effective compute depth.",
    "capital of france": "The capital of France is Paris.",
    "speed of light": "The speed of light in vacuum is 299,792,458 m/s.",
}

_ALLOWED_BINOPS = {
    ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
    ast.Div: _op.truediv, ast.Pow: _op.pow, ast.Mod: _op.mod,
    ast.FloorDiv: _op.floordiv,
}
_ALLOWED_UNARY = {ast.UAdd: _op.pos, ast.USub: _op.neg}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def get_weather(city: str) -> str:
    rec = _WEATHER.get(str(city).strip().lower())
    if not rec:
        return f"No weather data for {city!r}."
    return f"{rec['temp_c']}°C, {rec['conditions']}"


def calculator(expression: str) -> str:
    try:
        val = _safe_eval(ast.parse(str(expression), mode="eval"))
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"
    return str(val)


def web_search(query: str) -> str:
    key = str(query).strip().lower()
    for k, v in _SEARCH.items():
        if k in key:
            return v
    return f"No results found for {query!r}."


def get_current_time(timezone: str = "UTC") -> str:
    # Real time is fine at inference; the dataset hardcodes observations.
    now = _dt.datetime.now(_dt.timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M UTC")


# ---- registry --------------------------------------------------------------

TOOLS = {
    "get_weather": {
        "fn": get_weather,
        "schema": {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "City name"}},
                    "required": ["city"],
                },
            },
        },
    },
    "calculator": {
        "fn": calculator,
        "schema": {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate a basic arithmetic expression (+ - * / ** %).",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string", "description": "e.g. '3 * (4 + 5)'"}},
                    "required": ["expression"],
                },
            },
        },
    },
    "web_search": {
        "fn": web_search,
        "schema": {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for a factual query and return a short snippet.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query"}},
                    "required": ["query"],
                },
            },
        },
    },
    "get_current_time": {
        "fn": get_current_time,
        "schema": {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current date and time.",
                "parameters": {
                    "type": "object",
                    "properties": {"timezone": {"type": "string", "description": "IANA tz, default UTC"}},
                    "required": [],
                },
            },
        },
    },
}


def schema_for(name: str) -> dict:
    return TOOLS[name]["schema"]


def execute(name: str, arguments: dict) -> str:
    if name not in TOOLS:
        return f"error: unknown tool {name!r}"
    try:
        return str(TOOLS[name]["fn"](**(arguments or {})))
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"
