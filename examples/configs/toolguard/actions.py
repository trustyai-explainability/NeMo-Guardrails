import json
import os
from typing import Any, Dict, Tuple
import logging
import httpx

from typing import Optional
from nemoguardrails.actions import action
import logging

TOOLGUARD_URL = os.getenv("TOOLGUARD_URL", "http://host.containers.internal:8080")

log = logging.getLogger(__name__)


# output rails
def read_context(context: Any) -> Dict[str, Any]:
    """
    tool_message can be:
      - JSON string like: "{\"g\":10,\"h\":0}"
      - dict like: {"g":10,"h":0}
    """
    if isinstance(context, dict):
        return context

    if isinstance(context, str):
        s = context.strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            # If not JSON, pass raw for debugging (optional)
            return {"raw": context}

    return {}


@action(name="toolguard_check")
async def toolguard_check(tool_calls=None, context=None) -> Dict[str, Any]:
    """
    Calls ToolGuard policy-check endpoint:
      POST {TOOLGUARD_URL}/tools/{tool_name}
      body: args dict

    Expects HTTP 200 always with JSON:
      {"tool": "...", "allowed": true}
      or
      {"tool": "...", "allowed": false, "error": {"type": "...", "message": "..."}}

    Returns (always):
      {"allowed": <bool>, "reason": <str>, "toolguard": <optional>}
    """
    parsed_context = read_context(context)

    if tool_calls is None:
        log.critical(" TG> : tool_call still EMPTY, parsing context ")

    # Always return these keys (prevents missing-key issues in Colang)
    result: Dict[str, Any] = {
        "allowed": False,
        "reason": "Empty context",
        "toolguard": None,
    }
    tool_calls = parsed_context.get("tool_calls", []) if context else []
    if len(tool_calls) == 0:
        return result

    if len(tool_calls) > 1:
        log.critical(f" TG> : Just one tool call supported at time ")

    tool_call = tool_calls[0]

    log.critical(f" TG> : tool_call : {tool_call}")
    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {})
    try:
        timeout = httpx.Timeout(30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{TOOLGUARD_URL}/tools/{tool_name}", json=tool_args)

        if r.status_code != 200:
            result["allowed"] = False
            result["reason"] = f"ToolGuard HTTP {r.status_code}"
            result["toolguard"] = {"http_status": r.status_code, "body": r.text}
            log.critical(" TG> : tool=%s args=%s -> %s", tool_name, tool_args, result)
            return result

        resp = r.json()
        result["toolguard"] = resp

        raw_allowed = resp.get("allowed", False)
        allowed = (
            raw_allowed is True
            or raw_allowed == 1
            or (isinstance(raw_allowed, str) and raw_allowed.lower() == "true")
        )
        result["allowed"] = allowed

        if not allowed:
            err = resp.get("error") or {}
            result["reason"] = err.get("message") or "Denied by ToolGuard policy"

        log.critical(
            " TG> : toolguard_check: tool=%s args=%s -> %s",
            tool_name,
            tool_args,
            result,
        )
    except Exception as e:
        # Fail-closed, never crash NeMo
        result["allowed"] = False
        result["reason"] = f"ToolGuard call failed: {e!r}"
        log.critical(" TG> : toolguard_check failed")

    return result
