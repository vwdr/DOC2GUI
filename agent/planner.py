import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.llm import LLM


MAX_PROMPT_CHARS = 1800
MAX_CHUNK_CHARS = 240
MAX_EVIDENCE_CHUNKS = 4
MAX_USERDATA_CHARS = 400


@dataclass
class FieldInfo:
    field_id: str
    label: str
    field_type: str
    selector: str
    options: Optional[List[str]] = None


@dataclass
class ActionStep:
    action_type: str
    selector: str
    value: Optional[str]
    evidence: List[str]


def _truncate_text(text: str, max_chars: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3] + "..."


def _build_prompt(fields: List[FieldInfo], user_data: Dict[str, Any], retrieved_chunks: List[Dict[str, str]]) -> str:
    field_lines = []
    for field in fields:
        options = ", ".join(field.options or [])
        field_lines.append(f"- {field.field_id} ({field.field_type}) label='{field.label}' selector='{field.selector}' options='{options}'")

    chunk_lines = []
    for chunk in retrieved_chunks:
        chunk_text = _truncate_text(chunk["text"], MAX_CHUNK_CHARS)
        chunk_lines.append(f"[{chunk['chunk_id']}] {chunk_text}")

    user_data_str = json.dumps(user_data, indent=2)
    prompt = (
        "You are a form-filling planner. Output ONLY valid JSON.\n"
        "Use the retrieved evidence chunks to justify each action.\n"
        "JSON schema: {\"actions\": [{\"action_type\": \"fill|select|check|upload|submit\","
        " \"selector\": string, \"value\": string|null, \"evidence\": [chunk_id]}]}\n\n"
        "Available fields:\n"
        + "\n".join(field_lines)
        + "\n\nUser data JSON:\n"
        + user_data_str
        + "\n\nRetrieved evidence:\n"
        + "\n".join(chunk_lines)
        + "\n\nPlan the minimal steps to fill the form and submit."
    )
    while len(prompt) > MAX_PROMPT_CHARS and chunk_lines:
        chunk_lines.pop()
        prompt = (
            "You are a form-filling planner. Output ONLY valid JSON.\n"
            "Use the retrieved evidence chunks to justify each action.\n"
            "JSON schema: {\"actions\": [{\"action_type\": \"fill|select|check|upload|submit\","
            " \"selector\": string, \"value\": string|null, \"evidence\": [chunk_id]}]}\n\n"
            "Available fields:\n"
            + "\n".join(field_lines)
            + "\n\nUser data JSON:\n"
            + user_data_str
            + "\n\nRetrieved evidence:\n"
            + "\n".join(chunk_lines)
            + "\n\nPlan the minimal steps to fill the form and submit."
        )
    if len(prompt) > MAX_PROMPT_CHARS and len(user_data_str) > MAX_USERDATA_CHARS:
        user_data_str = _truncate_text(user_data_str, MAX_USERDATA_CHARS)
        prompt = (
            "You are a form-filling planner. Output ONLY valid JSON.\n"
            "Use the retrieved evidence chunks to justify each action.\n"
            "JSON schema: {\"actions\": [{\"action_type\": \"fill|select|check|upload|submit\","
            " \"selector\": string, \"value\": string|null, \"evidence\": [chunk_id]}]}\n\n"
            "Available fields:\n"
            + "\n".join(field_lines)
            + "\n\nUser data JSON:\n"
            + user_data_str
            + "\n\nRetrieved evidence:\n"
            + "\n".join(chunk_lines)
            + "\n\nPlan the minimal steps to fill the form and submit."
        )
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = _truncate_text(prompt, MAX_PROMPT_CHARS)
    return prompt


def _safe_json(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found")
    return json.loads(text[start : end + 1])


def plan_actions(llm: LLM, fields: List[FieldInfo], user_data: Dict[str, Any], retrieved_chunks: List[Dict[str, str]]) -> List[ActionStep]:
    trimmed_chunks = retrieved_chunks[:MAX_EVIDENCE_CHUNKS]
    prompt = _build_prompt(fields, user_data, trimmed_chunks)
    raw = llm.generate(prompt)
    try:
        payload = _safe_json(raw)
        actions = []
        default_evidence = [chunk["chunk_id"] for chunk in trimmed_chunks[:2]]
        for action in payload.get("actions", []):
            evidence = action.get("evidence", []) or default_evidence
            actions.append(
                ActionStep(
                    action_type=action.get("action_type", ""),
                    selector=action.get("selector", ""),
                    value=action.get("value"),
                    evidence=evidence,
                )
            )
        if actions:
            return actions
    except Exception:
        pass

    fallback_actions: List[ActionStep] = []
    for field in fields:
        if field.field_id not in user_data:
            continue
        value = user_data[field.field_id]
        if field.field_type == "checkbox":
            action_type = "check"
            value_str = "true" if str(value).lower() in {"true", "1", "yes"} else "false"
        elif field.field_type == "select":
            action_type = "select"
            value_str = str(value)
        elif field.field_type == "file":
            action_type = "upload"
            value_str = str(value)
        else:
            action_type = "fill"
            value_str = str(value)
        fallback_actions.append(
            ActionStep(
                action_type=action_type,
                selector=field.selector,
                value=value_str,
                evidence=[chunk["chunk_id"] for chunk in trimmed_chunks[:2]],
            )
        )
    fallback_actions.append(
        ActionStep(
            action_type="submit",
            selector="[data-field='submit']",
            value=None,
            evidence=[chunk["chunk_id"] for chunk in trimmed_chunks[:1]],
        )
    )
    return fallback_actions
