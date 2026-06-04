"""Write a small seed tool-use SFT dataset to data/tool_sft.jsonl.

This is a TEMPLATE / smoke-test set (~20 conversations). For a real overnight
run, scale this to thousands of examples — either by hand, by generating more
here, or by reformatting a public function-calling set (Glaive v2, Hermes
function-calling, Salesforce xLAM/APIGen) into the same conversation schema.

Conversation schema:
  {"tools": [tool names available],
   "user": "<user message>",
   "steps": [
     {"type":"tool_call","name":..,"arguments":{..},"observation":"<tool result>"},
     {"type":"final","content":"<assistant answer>"}
   ]}
Include some conversations whose only step is a "final" (no tool) so the model
learns to answer directly when no tool is warranted.
"""
import json
import os

from tools import schema_for

ALL = ["get_weather", "calculator", "web_search", "get_current_time"]

CONVOS = [
    # --- single tool call then answer ---
    {"tools": ALL, "user": "What's the weather in Paris right now?",
     "steps": [
         {"type": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"},
          "observation": "18°C, partly cloudy"},
         {"type": "final", "content": "It's currently 18°C and partly cloudy in Paris."}]},
    {"tools": ALL, "user": "How's the weather over in Tokyo?",
     "steps": [
         {"type": "tool_call", "name": "get_weather", "arguments": {"city": "Tokyo"},
          "observation": "23°C, clear"},
         {"type": "final", "content": "Tokyo is 23°C with clear skies."}]},
    {"tools": ALL, "user": "Is it raining in London?",
     "steps": [
         {"type": "tool_call", "name": "get_weather", "arguments": {"city": "London"},
          "observation": "14°C, light rain"},
         {"type": "final", "content": "Yes — London has light rain and is about 14°C."}]},
    {"tools": ALL, "user": "What is 23 times 47?",
     "steps": [
         {"type": "tool_call", "name": "calculator", "arguments": {"expression": "23 * 47"},
          "observation": "1081"},
         {"type": "final", "content": "23 × 47 = 1081."}]},
    {"tools": ALL, "user": "Compute (128 + 64) / 8 for me.",
     "steps": [
         {"type": "tool_call", "name": "calculator", "arguments": {"expression": "(128 + 64) / 8"},
          "observation": "24.0"},
         {"type": "final", "content": "(128 + 64) / 8 = 24."}]},
    {"tools": ALL, "user": "What's 2 to the power of 10?",
     "steps": [
         {"type": "tool_call", "name": "calculator", "arguments": {"expression": "2 ** 10"},
          "observation": "1024"},
         {"type": "final", "content": "2^10 = 1024."}]},
    {"tools": ALL, "user": "Look up the speed of light for me.",
     "steps": [
         {"type": "tool_call", "name": "web_search", "arguments": {"query": "speed of light"},
          "observation": "The speed of light in vacuum is 299,792,458 m/s."},
         {"type": "final", "content": "The speed of light in vacuum is 299,792,458 m/s."}]},
    {"tools": ALL, "user": "Search for what the Hierarchical Reasoning Model is.",
     "steps": [
         {"type": "tool_call", "name": "web_search",
          "arguments": {"query": "Hierarchical Reasoning Model"},
          "observation": "HRM is a recurrent architecture with two coupled modules "
                         "(slow high-level planner, fast low-level worker)."},
         {"type": "final", "content": "The Hierarchical Reasoning Model (HRM) is a recurrent "
                                      "architecture with two coupled modules: a slow high-level "
                                      "planner and a fast low-level worker."}]},
    {"tools": ALL, "user": "What time is it?",
     "steps": [
         {"type": "tool_call", "name": "get_current_time", "arguments": {},
          "observation": "2026-06-03 14:05 UTC"},
         {"type": "final", "content": "It's currently 14:05 UTC on 2026-06-03."}]},
    {"tools": ALL, "user": "What's the weather in San Francisco?",
     "steps": [
         {"type": "tool_call", "name": "get_weather", "arguments": {"city": "San Francisco"},
          "observation": "16°C, foggy"},
         {"type": "final", "content": "San Francisco is 16°C and foggy right now."}]},

    # --- multi-step: two tool calls ---
    {"tools": ALL, "user": "What's the weather in New York, and what's 15% of 200?",
     "steps": [
         {"type": "tool_call", "name": "get_weather", "arguments": {"city": "New York"},
          "observation": "21°C, sunny"},
         {"type": "tool_call", "name": "calculator", "arguments": {"expression": "0.15 * 200"},
          "observation": "30.0"},
         {"type": "final", "content": "New York is 21°C and sunny, and 15% of 200 is 30."}]},
    {"tools": ALL, "user": "Add the temperatures in Paris and London together.",
     "steps": [
         {"type": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"},
          "observation": "18°C, partly cloudy"},
         {"type": "tool_call", "name": "get_weather", "arguments": {"city": "London"},
          "observation": "14°C, light rain"},
         {"type": "tool_call", "name": "calculator", "arguments": {"expression": "18 + 14"},
          "observation": "32"},
         {"type": "final", "content": "Paris is 18°C and London is 14°C, so together that's 32°C."}]},

    # --- no tool needed: answer directly (teach restraint) ---
    {"tools": ALL, "user": "Hi there! Can you help me today?",
     "steps": [
         {"type": "final", "content": "Hi! Absolutely — I can check the weather, do calculations, "
                                      "search the web, or tell you the time. What do you need?"}]},
    {"tools": ALL, "user": "Thanks, that's all for now.",
     "steps": [
         {"type": "final", "content": "You're welcome! Reach out anytime you need a hand."}]},
    {"tools": ALL, "user": "What kinds of things can you do?",
     "steps": [
         {"type": "final", "content": "I can get the current weather for a city, evaluate "
                                      "arithmetic, search the web for facts, and report the "
                                      "current time."}]},
    {"tools": ALL, "user": "Who are you?",
     "steps": [
         {"type": "final", "content": "I'm a function-calling assistant that can use a few tools "
                                      "to help answer your questions."}]},
]


def normalize(c):
    """Convert a readable seed conversation into the on-disk multi-turn schema:
    tool names -> full schemas; typed steps -> user/calls/final turns."""
    turns = [{"role": "user", "content": c["user"]}]
    for s in c["steps"]:
        if s["type"] == "tool_call":
            turns.append({"role": "calls", "calls": [{"name": s["name"], "arguments": s["arguments"]}],
                          "observations": [s["observation"]]})
        else:
            turns.append({"role": "final", "content": s["content"]})
    return {"tools": [schema_for(n) for n in c["tools"]], "turns": turns}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "data")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "tool_sft.jsonl")
    with open(out, "w") as f:
        for c in CONVOS:
            f.write(json.dumps(normalize(c)) + "\n")
    print(f"wrote {len(CONVOS)} conversations -> {out}")


if __name__ == "__main__":
    main()
