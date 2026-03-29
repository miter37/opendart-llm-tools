from __future__ import annotations
import datetime as dt
import json
import os
import queue
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from opendart_llm_tools import create_dart_agent, dart_llm_tools_gemini, dart_llm_tools_openai
from opendart_llm_tools.opendart_llm_tools import (
    GEMINI_DEFAULT_MODEL,
    OPENAI_DEFAULT_MODEL,
    _load_key_env,
    _resolve_gemini_api_keys,
)


HTML_PATH = BASE_DIR / "test_console.html"
CSS_PATH = BASE_DIR / "web_test_console.css"
JS_PATH = BASE_DIR / "web_test_console.js"

APP_TITLE = "opendart_llm_tools"
OPENAI_CHAT_MODEL = OPENAI_DEFAULT_MODEL
OPENAI_DART_TOOL_MODEL = OPENAI_DEFAULT_MODEL
GEMINI_CHAT_MODEL = GEMINI_DEFAULT_MODEL
GEMINI_DART_TOOL_MODEL = GEMINI_DEFAULT_MODEL
GEMINI_CHAT_THINKING_LEVEL = "high"

PROVIDER_CONFIG: Dict[str, Dict[str, str]] = {
    "gemini": {
        "label": "Gemini",
        "chat_model": GEMINI_CHAT_MODEL,
        "dart_tool_model": GEMINI_DART_TOOL_MODEL,
    },
    "openai": {
        "label": "OpenAI",
        "chat_model": OPENAI_CHAT_MODEL,
        "dart_tool_model": OPENAI_DART_TOOL_MODEL,
    },
}

PUBLIC_TOOL_SYSTEM_PROMPT = (
    "You are a Korean assistant using an optional DART retrieval tool. "
    "Decide yourself whether the tool is necessary. "
    "Use find_dart_material only when the answer should be grounded in Korean DART disclosures. "
    "If the question is general knowledge, casual conversation, opinion, or not about Korean DART filings, "
    "answer directly without calling the tool. "
    "If you use the tool, rely on the returned material and source_paths for filing-grounded facts. "
    "Do not pretend to have used DART when you did not call the tool."
)


def _public_tool_system_prompt() -> str:
    today = dt.date.today().isoformat()
    return (
        f"{PUBLIC_TOOL_SYSTEM_PROMPT} "
        f"Today's actual date is {today}. "
        "If the user uses relative time such as recent 6 months, recent 3 months, recent 1 year, this year, last year, latest, or recent, "
        "do not rewrite it into a guessed absolute date range in the tool query. "
        "Preserve the user's relative time expression and let the DART tool interpret it against the actual runtime date. "
        "If the user gives an explicit date or explicit period, preserve it exactly."
    )

PUBLIC_TOOL_FINAL_PROMPT = (
    "Answer the user in Korean. "
    "If tool results are present, use them as the DART-grounded evidence. "
    "If the tool failed or found insufficient material, state that clearly instead of inventing unsupported facts. "
    "Do not expose raw JSON unless it helps the user."
)

PUBLIC_TOOL_FOLLOWUP_PROMPT = (
    "You are deciding whether one additional DART tool request is needed before giving the final answer. "
    "After receiving the first DART materials, decide whether they are sufficient or whether one more tool request is needed. "
    "If the first materials are sufficient, stop and do not call the tool again. "
    "If they are still insufficient, you may call the tool exactly one more time. "
    "When making that additional request, think carefully about whether you should ask for: "
    "material from another company and another period, "
    "material from the same company but a different period or different filing, "
    "or material from multiple companies and periods if the user request truly requires that. "
    "Use the additional tool request only when it can materially improve the final answer. "
    "This is the last tool round. After this decision, the final answer must be based on everything gathered so far."
)


class RunRequest(BaseModel):
    provider: str
    question: str


def _json_text(payload: Any, limit: int = 6000) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception:
        text = str(payload)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (truncated)"


def _push_log(
    logs: List[Dict[str, str]],
    title: str,
    detail: Any,
    *,
    kind: str = "info",
    event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    entry = {
        "kind": kind,
        "title": title,
        "detail": detail if isinstance(detail, str) else _json_text(detail),
    }
    logs.append(entry)
    if event_callback:
        event_callback({"type": "log", "log": entry})


def _sanitize_text(text: Any) -> str:
    return str(text or "").strip()


def _empty_material_result() -> Dict[str, Any]:
    return {"ok": False, "text": "", "source_paths": [], "error": ""}


def _merge_material_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return _empty_material_result()

    merged_texts: List[str] = []
    source_paths: List[str] = []
    seen_paths = set()
    errors: List[str] = []

    for result in results:
        if not isinstance(result, dict):
            continue
        if result.get("ok"):
            text = _sanitize_text(result.get("text"))
            if text:
                merged_texts.append(text)
            for path in result.get("source_paths") or []:
                path_text = _sanitize_text(path)
                if path_text and path_text not in seen_paths:
                    seen_paths.add(path_text)
                    source_paths.append(path_text)
        elif result.get("error"):
            errors.append(str(result.get("error")))

    if merged_texts or source_paths:
        return {
            "ok": True,
            "text": "\n\n".join(merged_texts).strip(),
            "source_paths": source_paths,
            "error": "",
        }

    return {
        "ok": False,
        "text": "",
        "source_paths": [],
        "error": errors[0] if errors else "",
    }


def _bind_internal_logging(
    agent: Any,
    *,
    logs: List[Dict[str, str]],
    event_callback: Optional[Callable[[Dict[str, Any]], None]],
) -> Any:
    def push_log(title: str, detail: Any, kind: str = "info") -> None:
        _push_log(logs, title, detail, kind=kind, event_callback=event_callback)

    original_execute_tool = agent.execute_tool

    def execute_tool_logged(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        push_log(f"Internal tool call: {name}", args, kind="tool-call")
        result = original_execute_tool(name, args)
        push_log(f"Internal tool result: {name}", result, kind="tool-result")
        return result

    setattr(agent, "push_log", push_log)
    setattr(agent, "execute_tool", execute_tool_logged)
    return agent


def _run_public_tool_with_internal_logs(
    *,
    name: str,
    args: Dict[str, Any],
    provider: str,
    model: str,
    dart_api_key: str,
    openai_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    fallback_api_keys: Optional[List[str]] = None,
    key_env: Optional[Dict[str, str]] = None,
    logs: List[Dict[str, str]],
    event_callback: Optional[Callable[[Dict[str, Any]], None]],
) -> Dict[str, Any]:
    agent = create_dart_agent(
        provider=provider,
        model=model,
        dart_api_key=dart_api_key,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        fallback_api_keys=fallback_api_keys,
        key_env=key_env,
    )
    agent = _bind_internal_logging(agent, logs=logs, event_callback=event_callback)
    return agent.execute_public_tool(name, args)


def _execute_public_tool_calls(
    *,
    calls: List[Dict[str, Any]],
    provider: str,
    model: str,
    dart_api_key: str,
    openai_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    fallback_api_keys: Optional[List[str]] = None,
    key_env: Optional[Dict[str, str]] = None,
    logs: List[Dict[str, str]],
    event_callback: Optional[Callable[[Dict[str, Any]], None]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    tool_records: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []
    for call in calls:
        name = str(call.get("name", "") or "")
        args = call.get("args", {}) or {}
        _push_log(
            logs,
            f"Public tool call: {name}",
            args,
            kind="tool-call",
            event_callback=event_callback,
        )
        result = _run_public_tool_with_internal_logs(
            name=name,
            args=args,
            provider=provider,
            model=model,
            dart_api_key=dart_api_key,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            fallback_api_keys=fallback_api_keys,
            logs=logs,
            event_callback=event_callback,
            key_env=key_env,
        )
        _push_log(
            logs,
            f"Public tool result: {name}",
            result,
            kind="tool-result",
            event_callback=event_callback,
        )
        tool_results.append(result)
        tool_records.append(
            {
                "tool_name": name,
                "arguments": args,
                "result": result,
            }
        )
    return tool_records, tool_results


@lru_cache(maxsize=1)
def get_key_env() -> Dict[str, str]:
    return _load_key_env(str(PROJECT_ROOT / "key.env"))


def _get_dart_api_key(key_env: Dict[str, str]) -> str:
    dart_key = os.environ.get("DART_API_KEY") or key_env.get("opendart_key")
    if not dart_key:
        raise RuntimeError("DART_API_KEY/opendart_key not found")
    return dart_key


def _get_openai_api_key(key_env: Dict[str, str]) -> str:
    openai_key = os.environ.get("OPENAI_API_KEY") or key_env.get("openai_key")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY/openai_key not found")
    return openai_key


def _get_gemini_api_keys(key_env: Dict[str, str]) -> List[str]:
    gemini_keys = _resolve_gemini_api_keys(key_env)
    if not gemini_keys:
        raise RuntimeError("Gemini API key not found")
    return gemini_keys


def _openai_function_calls(response: Any) -> List[Any]:
    output = getattr(response, "output", None) or []
    return [item for item in output if getattr(item, "type", "") == "function_call"]


def _openai_output_types(response: Any) -> List[str]:
    output = getattr(response, "output", None) or []
    return [str(getattr(item, "type", "")) for item in output]


def _gemini_generate_content(
    *,
    api_keys: List[str],
    model: str,
    payload: Dict[str, Any],
    timeout: int = 60,
) -> Tuple[Dict[str, Any], str]:
    api_url = "https://generativelanguage.googleapis.com/v1beta/models"
    last_error: Optional[Exception] = None
    max_attempts = 3
    retry_delay_seconds = 3

    for api_key in api_keys:
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.post(
                    f"{api_url}/{model}:generateContent",
                    params={"key": api_key},
                    json=payload,
                    timeout=timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds)
                    continue
                break
            if response.ok:
                return response.json(), api_key
            last_error = requests.HTTPError(
                f"{response.status_code} Error: {response.text}",
                response=response,
            )
            if response.status_code not in {403, 429, 500, 503, 504}:
                raise last_error
            if attempt < max_attempts:
                time.sleep(retry_delay_seconds)

    if last_error:
        raise last_error
    raise RuntimeError("Gemini request failed without a response")


def _gemini_content(payload: Dict[str, Any]) -> Dict[str, Any]:
    candidates = payload.get("candidates") or []
    if not candidates:
        return {}
    return (candidates[0] or {}).get("content") or {}


def _gemini_text(content: Dict[str, Any]) -> str:
    texts = [part.get("text", "") for part in content.get("parts") or [] if "text" in part]
    return _sanitize_text("\n".join(texts))


def _gemini_function_calls(content: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [part.get("functionCall", {}) for part in content.get("parts") or [] if "functionCall" in part]


def _openai_final_answer(
    *,
    question: str,
    tool_records: List[Dict[str, Any]],
    chat_model: str,
    openai_api_key: str,
) -> str:
    client = OpenAI(api_key=openai_api_key)
    prompt = (
        f"{PUBLIC_TOOL_FINAL_PROMPT}\n\n"
        f"User request:\n{question}\n\n"
        "Tool results:\n"
        f"{_json_text(tool_records, limit=20000)}"
    )
    response = client.responses.create(
        model=chat_model,
        instructions=PUBLIC_TOOL_FINAL_PROMPT,
        input=prompt,
    )
    return _sanitize_text(getattr(response, "output_text", ""))


def _gemini_final_answer(
    *,
    question: str,
    tool_records: List[Dict[str, Any]],
    chat_model: str,
    api_keys: List[str],
) -> str:
    prompt = (
        f"{PUBLIC_TOOL_FINAL_PROMPT}\n\n"
        f"User request:\n{question}\n\n"
        "Tool results:\n"
        f"{_json_text(tool_records, limit=20000)}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": PUBLIC_TOOL_FINAL_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "thinkingConfig": {"thinkingLevel": GEMINI_CHAT_THINKING_LEVEL},
        },
    }
    response_json, _ = _gemini_generate_content(
        api_keys=api_keys,
        model=chat_model,
        payload=payload,
    )
    return _gemini_text(_gemini_content(response_json))


def _run_openai_lookup(
    question: str,
    *,
    logs: List[Dict[str, str]],
    event_callback: Optional[Callable[[Dict[str, Any]], None]],
) -> Dict[str, Any]:
    key_env = get_key_env()
    dart_key = _get_dart_api_key(key_env)
    openai_key = _get_openai_api_key(key_env)
    chat_model = OPENAI_CHAT_MODEL
    dart_tool_model = OPENAI_DART_TOOL_MODEL

    if OpenAI is None:
        raise RuntimeError("openai package is not available")

    client = OpenAI(api_key=openai_key)
    _push_log(logs, "Provider", "OpenAI", kind="meta", event_callback=event_callback)
    _push_log(logs, "Question", question, kind="meta", event_callback=event_callback)
    _push_log(
        logs,
        "Model stack",
        {
            "chat_model": chat_model,
            "dart_tool_model": dart_tool_model,
        },
        kind="meta",
        event_callback=event_callback,
    )
    _push_log(
        logs,
        "LLM_A request",
        {
            "phase": "tool-decision",
            "model": chat_model,
            "tool_count": len(dart_llm_tools_openai()),
        },
        kind="model",
        event_callback=event_callback,
    )
    response = client.responses.create(
        model=chat_model,
        instructions=_public_tool_system_prompt(),
        input=question,
        tools=dart_llm_tools_openai(),
    )
    function_calls = _openai_function_calls(response)
    _push_log(
        logs,
        "LLM_A response",
        {
            "phase": "tool-decision",
            "model": chat_model,
            "output_types": _openai_output_types(response),
            "function_calls": [
                {
                    "name": getattr(item, "name", ""),
                    "call_id": getattr(item, "call_id", ""),
                }
                for item in function_calls
            ],
            "output_text": _sanitize_text(getattr(response, "output_text", ""))[:4000],
        },
        kind="llm-response",
        event_callback=event_callback,
    )

    if not function_calls:
        return {
            "ok": True,
            "provider": "openai",
            "chat_model": chat_model,
            "dart_tool_model": dart_tool_model,
            "tool_used": False,
            "result": _empty_material_result(),
            "answer": _sanitize_text(getattr(response, "output_text", "")),
            "source_paths": [],
            "error": "",
        }

    initial_calls: List[Dict[str, Any]] = []
    for item in function_calls:
        raw_arguments = getattr(item, "arguments", "") or "{}"
        try:
            args = json.loads(raw_arguments)
        except Exception:
            args = {}
        initial_calls.append({"name": getattr(item, "name", ""), "args": args})

    tool_records, tool_results = _execute_public_tool_calls(
        calls=initial_calls,
        provider="openai",
        model=dart_tool_model,
        dart_api_key=dart_key,
        openai_api_key=openai_key,
        logs=logs,
        event_callback=event_callback,
        key_env=key_env,
    )

    material_result = _merge_material_results(tool_results)
    _push_log(logs, "Material result", material_result, kind="final", event_callback=event_callback)
    _push_log(
        logs,
        "LLM_A request",
        {
            "phase": "followup-decision",
            "model": chat_model,
            "tool_round": 2,
            "max_tool_rounds": 2,
        },
        kind="model",
        event_callback=event_callback,
    )
    followup_prompt = (
        f"{PUBLIC_TOOL_FOLLOWUP_PROMPT}\n\n"
        f"User request:\n{question}\n\n"
        "First-round tool results:\n"
        f"{_json_text(tool_records, limit=20000)}"
    )
    followup_response = client.responses.create(
        model=chat_model,
        instructions=PUBLIC_TOOL_FOLLOWUP_PROMPT,
        input=followup_prompt,
        tools=dart_llm_tools_openai(),
    )
    followup_calls_raw = _openai_function_calls(followup_response)
    followup_calls: List[Dict[str, Any]] = []
    for item in followup_calls_raw:
        raw_arguments = getattr(item, "arguments", "") or "{}"
        try:
            args = json.loads(raw_arguments)
        except Exception:
            args = {}
        followup_calls.append({"name": getattr(item, "name", ""), "args": args})
    _push_log(
        logs,
        "LLM_A response",
        {
            "phase": "followup-decision",
            "model": chat_model,
            "function_calls": followup_calls,
            "output_text": _sanitize_text(getattr(followup_response, "output_text", ""))[:4000],
        },
        kind="llm-response",
        event_callback=event_callback,
    )
    if followup_calls:
        followup_tool_records, followup_tool_results = _execute_public_tool_calls(
            calls=followup_calls,
            provider="openai",
            model=dart_tool_model,
            dart_api_key=dart_key,
            openai_api_key=openai_key,
            logs=logs,
            event_callback=event_callback,
            key_env=key_env,
        )
        tool_records.extend(followup_tool_records)
        tool_results.extend(followup_tool_results)
        material_result = _merge_material_results(tool_results)
        _push_log(logs, "Material result", material_result, kind="final", event_callback=event_callback)
    _push_log(
        logs,
        "LLM_A request",
        {
            "phase": "final-answer",
            "model": chat_model,
            "tool_used": True,
        },
        kind="model",
        event_callback=event_callback,
    )
    answer = _openai_final_answer(
        question=question,
        tool_records=tool_records,
        chat_model=chat_model,
        openai_api_key=openai_key,
    )
    return {
        "ok": True,
        "provider": "openai",
        "chat_model": chat_model,
        "dart_tool_model": dart_tool_model,
        "tool_used": True,
        "result": material_result,
        "answer": answer,
        "source_paths": material_result.get("source_paths", []),
        "error": material_result.get("error", ""),
    }


def _run_gemini_lookup(
    question: str,
    *,
    logs: List[Dict[str, str]],
    event_callback: Optional[Callable[[Dict[str, Any]], None]],
) -> Dict[str, Any]:
    key_env = get_key_env()
    dart_key = _get_dart_api_key(key_env)
    gemini_keys = _get_gemini_api_keys(key_env)
    chat_model = GEMINI_CHAT_MODEL
    dart_tool_model = GEMINI_DART_TOOL_MODEL

    _push_log(logs, "Provider", "Gemini", kind="meta", event_callback=event_callback)
    _push_log(logs, "Question", question, kind="meta", event_callback=event_callback)
    _push_log(
        logs,
        "Model stack",
        {
            "chat_model": chat_model,
            "dart_tool_model": dart_tool_model,
        },
        kind="meta",
        event_callback=event_callback,
    )
    _push_log(
        logs,
        "LLM_A request",
        {
            "phase": "tool-decision",
            "model": chat_model,
            "thinking_level": GEMINI_CHAT_THINKING_LEVEL,
            "tool_count": len(dart_llm_tools_gemini()),
        },
        kind="model",
        event_callback=event_callback,
    )

    payload = {
        "systemInstruction": {"parts": [{"text": _public_tool_system_prompt()}]},
        "contents": [{"role": "user", "parts": [{"text": question}]}],
        "tools": dart_llm_tools_gemini(),
        "toolConfig": {
            "functionCallingConfig": {
                "mode": "AUTO",
            }
        },
        "generationConfig": {
            "thinkingConfig": {"thinkingLevel": GEMINI_CHAT_THINKING_LEVEL},
        },
    }
    response_json, _ = _gemini_generate_content(
        api_keys=gemini_keys,
        model=chat_model,
        payload=payload,
    )
    content = _gemini_content(response_json)
    function_calls = _gemini_function_calls(content)
    _push_log(
        logs,
        "LLM_A response",
        {
            "phase": "tool-decision",
            "model": chat_model,
            "function_calls": function_calls,
            "output_text": _gemini_text(content)[:4000],
        },
        kind="llm-response",
        event_callback=event_callback,
    )

    if not function_calls:
        return {
            "ok": True,
            "provider": "gemini",
            "chat_model": chat_model,
            "dart_tool_model": dart_tool_model,
            "tool_used": False,
            "result": _empty_material_result(),
            "answer": _gemini_text(content),
            "source_paths": [],
            "error": "",
        }

    initial_calls = [
        {
            "name": str(function_call.get("name", "") or ""),
            "args": function_call.get("args", {}) or {},
        }
        for function_call in function_calls
    ]
    tool_records, tool_results = _execute_public_tool_calls(
        calls=initial_calls,
        provider="gemini",
        model=dart_tool_model,
        dart_api_key=dart_key,
        gemini_api_key=gemini_keys[0],
        fallback_api_keys=gemini_keys[1:],
        logs=logs,
        event_callback=event_callback,
        key_env=key_env,
    )

    material_result = _merge_material_results(tool_results)
    _push_log(logs, "Material result", material_result, kind="final", event_callback=event_callback)
    _push_log(
        logs,
        "LLM_A request",
        {
            "phase": "followup-decision",
            "model": chat_model,
            "thinking_level": GEMINI_CHAT_THINKING_LEVEL,
            "tool_round": 2,
            "max_tool_rounds": 2,
        },
        kind="model",
        event_callback=event_callback,
    )
    followup_payload = {
        "systemInstruction": {"parts": [{"text": PUBLIC_TOOL_FOLLOWUP_PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            f"{PUBLIC_TOOL_FOLLOWUP_PROMPT}\n\n"
                            f"User request:\n{question}\n\n"
                            "First-round tool results:\n"
                            f"{_json_text(tool_records, limit=20000)}"
                        )
                    }
                ],
            }
        ],
        "tools": dart_llm_tools_gemini(),
        "toolConfig": {
            "functionCallingConfig": {
                "mode": "AUTO",
            }
        },
        "generationConfig": {
            "thinkingConfig": {"thinkingLevel": GEMINI_CHAT_THINKING_LEVEL},
        },
    }
    followup_response_json, _ = _gemini_generate_content(
        api_keys=gemini_keys,
        model=chat_model,
        payload=followup_payload,
    )
    followup_content = _gemini_content(followup_response_json)
    followup_calls = _gemini_function_calls(followup_content)
    _push_log(
        logs,
        "LLM_A response",
        {
            "phase": "followup-decision",
            "model": chat_model,
            "function_calls": followup_calls,
            "output_text": _gemini_text(followup_content)[:4000],
        },
        kind="llm-response",
        event_callback=event_callback,
    )
    if followup_calls:
        followup_tool_records, followup_tool_results = _execute_public_tool_calls(
            calls=[
                {
                    "name": str(function_call.get("name", "") or ""),
                    "args": function_call.get("args", {}) or {},
                }
                for function_call in followup_calls
            ],
            provider="gemini",
            model=dart_tool_model,
            dart_api_key=dart_key,
            gemini_api_key=gemini_keys[0],
            fallback_api_keys=gemini_keys[1:],
            logs=logs,
            event_callback=event_callback,
            key_env=key_env,
        )
        tool_records.extend(followup_tool_records)
        tool_results.extend(followup_tool_results)
        material_result = _merge_material_results(tool_results)
        _push_log(logs, "Material result", material_result, kind="final", event_callback=event_callback)
    _push_log(
        logs,
        "LLM_A request",
        {
            "phase": "final-answer",
            "model": chat_model,
            "thinking_level": GEMINI_CHAT_THINKING_LEVEL,
            "tool_used": True,
        },
        kind="model",
        event_callback=event_callback,
    )
    answer = _gemini_final_answer(
        question=question,
        tool_records=tool_records,
        chat_model=chat_model,
        api_keys=gemini_keys,
    )
    return {
        "ok": True,
        "provider": "gemini",
        "chat_model": chat_model,
        "dart_tool_model": dart_tool_model,
        "tool_used": True,
        "result": material_result,
        "answer": answer,
        "source_paths": material_result.get("source_paths", []),
        "error": material_result.get("error", ""),
    }


def _run_lookup(
    provider: str,
    question: str,
    *,
    logs: List[Dict[str, str]],
    event_callback: Optional[Callable[[Dict[str, Any]], None]],
) -> Dict[str, Any]:
    provider_norm = str(provider or "").strip().lower()
    if provider_norm == "openai":
        return _run_openai_lookup(question, logs=logs, event_callback=event_callback)
    if provider_norm == "gemini":
        return _run_gemini_lookup(question, logs=logs, event_callback=event_callback)
    raise RuntimeError("provider must be 'gemini' or 'openai'")


app = FastAPI(title=APP_TITLE)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(HTML_PATH)


@app.get("/assets/test-console.css")
def app_css() -> FileResponse:
    return FileResponse(CSS_PATH)


@app.get("/assets/test-console.js")
def app_js() -> FileResponse:
    return FileResponse(JS_PATH)


@app.get("/api/config")
def app_config() -> Dict[str, Any]:
    return {
        "title": APP_TITLE,
        "providers": PROVIDER_CONFIG,
    }


@app.post("/api/run")
def run_agent(body: RunRequest) -> Dict[str, Any]:
    question = _sanitize_text(body.question)
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    logs: List[Dict[str, str]] = []
    try:
        payload = _run_lookup(
            body.provider,
            question,
            logs=logs,
            event_callback=None,
        )
        return {
            **payload,
            "logs": logs,
        }
    except Exception as exc:
        logs.append(
            {
                "kind": "error",
                "title": "Execution error",
                "detail": str(exc),
            }
        )
        return {
            "ok": False,
            "provider": body.provider,
            "chat_model": PROVIDER_CONFIG.get(body.provider, {}).get("chat_model", ""),
            "dart_tool_model": PROVIDER_CONFIG.get(body.provider, {}).get("dart_tool_model", ""),
            "tool_used": False,
            "logs": logs,
            "result": {"ok": False, "text": "", "source_paths": [], "error": str(exc)},
            "answer": "",
            "source_paths": [],
            "error": str(exc),
        }


@app.post("/api/run/stream")
def run_agent_stream(body: RunRequest) -> StreamingResponse:
    question = _sanitize_text(body.question)
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    log_queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
    done = object()

    def emit(event: Dict[str, Any]) -> None:
        log_queue.put(event)

    def worker() -> None:
        logs: List[Dict[str, str]] = []
        try:
            payload = _run_lookup(
                body.provider,
                question,
                logs=logs,
                event_callback=emit,
            )
            emit({"type": "result", **payload})
        except Exception as exc:
            emit(
                {
                    "type": "log",
                    "log": {
                        "kind": "error",
                        "title": "Execution error",
                        "detail": str(exc),
                    },
                }
            )
            emit(
                {
                    "type": "result",
                    "ok": False,
                    "provider": body.provider,
                    "chat_model": PROVIDER_CONFIG.get(body.provider, {}).get("chat_model", ""),
                    "dart_tool_model": PROVIDER_CONFIG.get(body.provider, {}).get("dart_tool_model", ""),
                    "tool_used": False,
                    "result": {"ok": False, "text": "", "source_paths": [], "error": str(exc)},
                    "answer": "",
                    "source_paths": [],
                    "error": str(exc),
                }
            )
        finally:
            log_queue.put(done)

    threading.Thread(target=worker, daemon=True).start()

    def generate():
        while True:
            item = log_queue.get()
            if item is done:
                break
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")
