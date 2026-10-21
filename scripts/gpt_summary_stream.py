"""Accumulate only OpenAI's actual reasoning-summary stream events."""
from collections import defaultdict


def stream_result(events: list[dict]) -> tuple[dict | None, str]:
    response = None
    chunks = defaultdict(list)
    completed_parts = {}
    for event in events:
        kind = event.get("type")
        if kind in ("response.completed", "response.incomplete", "response.failed"):
            response = event.get("response")
        key = (event.get("item_id"), event.get("summary_index", 0))
        if kind == "response.reasoning_summary_text.delta" and isinstance(event.get("delta"), str):
            chunks[key].append(event["delta"])
        elif kind == "response.reasoning_summary_text.done" and isinstance(event.get("text"), str):
            completed_parts[key] = event["text"]
    keys = list(dict.fromkeys([*chunks, *completed_parts]))
    text = "\n\n".join(completed_parts.get(key, "".join(chunks[key])) for key in keys).strip()
    return response, text
