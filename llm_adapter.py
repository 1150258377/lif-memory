from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


PROVIDER_PRESETS = {
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "api_key_env": "DASHSCOPE_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-pro",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
        "api_key_env": "MOONSHOT_API_KEY",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
        "api_key_env": "ZHIPUAI_API_KEY",
    },
}

DEFAULT_LOCAL_CONFIG = Path(__file__).resolve().parent / "config" / "llm.local.json"


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    base_url: str
    model: str
    api_key: str
    timeout_seconds: int = 60
    temperature: float = 0.1


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--llm-review",
        action="store_true",
        help="Ask an OpenAI-compatible LLM to review top spike semantics without changing LIF decisions.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=[*PROVIDER_PRESETS.keys(), "custom"],
        default=os.environ.get("LIF_LLM_PROVIDER"),
        help="LLM provider preset.",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=os.environ.get("LIF_LLM_BASE_URL"),
        help="OpenAI-compatible base URL. Required for custom provider.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=os.environ.get("LIF_LLM_MODEL"),
        help="LLM model name. Defaults to provider preset.",
    )
    parser.add_argument(
        "--llm-api-key-env",
        type=str,
        default=os.environ.get("LIF_LLM_API_KEY_ENV"),
        help="Environment variable containing API key.",
    )
    parser.add_argument(
        "--llm-local-config",
        type=Path,
        default=Path(os.environ["LIF_LLM_LOCAL_CONFIG"]) if os.environ.get("LIF_LLM_LOCAL_CONFIG") else DEFAULT_LOCAL_CONFIG,
        help="Ignored local JSON config for API keys. Defaults to config/llm.local.json.",
    )
    parser.add_argument("--llm-timeout", type=int, default=60, help="LLM request timeout in seconds.")


def load_local_config(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def local_api_key(local_config: dict[str, object], provider: str) -> str:
    api_keys = local_config.get("api_keys")
    if isinstance(api_keys, dict) and isinstance(api_keys.get(provider), str):
        return str(api_keys[provider])
    provider_config = local_config.get(provider)
    if isinstance(provider_config, dict) and isinstance(provider_config.get("api_key"), str):
        return str(provider_config["api_key"])
    if isinstance(local_config.get("api_key"), str):
        return str(local_config["api_key"])
    return ""


def config_from_args(args: argparse.Namespace) -> LLMConfig:
    local_config = load_local_config(getattr(args, "llm_local_config", DEFAULT_LOCAL_CONFIG))
    provider_arg = getattr(args, "llm_provider", None)
    provider = str(provider_arg or local_config.get("provider") or "qwen")
    preset = PROVIDER_PRESETS.get(provider, {})
    provider_config = local_config.get(provider)
    if not isinstance(provider_config, dict):
        provider_config = {}
    use_local_default_model = not provider_arg or provider == local_config.get("provider")
    base_url = str(args.llm_base_url or provider_config.get("base_url") or local_config.get("base_url") or preset.get("base_url", "")).rstrip("/")
    model = str(
        args.llm_model
        or provider_config.get("model")
        or (local_config.get("model") if use_local_default_model else None)
        or preset.get("model", "")
    )
    api_key_env = str(args.llm_api_key_env or preset.get("api_key_env", "LIF_LLM_API_KEY"))
    api_key = os.environ.get(api_key_env, "") or local_api_key(local_config, provider)
    if not base_url:
        raise SystemExit("LLM base URL is required. Use --llm-base-url or a provider preset.")
    if not model:
        raise SystemExit("LLM model is required. Use --llm-model or a provider preset.")
    if not api_key:
        raise SystemExit(f"Missing LLM API key. Set environment variable {api_key_env}.")
    return LLMConfig(
        provider=provider,
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_seconds=int(args.llm_timeout),
    )


def extract_json_object(text: str) -> dict[str, object]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {"raw": data}
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        data = json.loads(cleaned[start : end + 1])
        return data if isinstance(data, dict) else {"raw": data}
    raise ValueError("LLM response did not contain a JSON object.")


def review_prompt(spike_packet: dict[str, object], allowed_states: list[str]) -> list[dict[str, str]]:
    compact_packet = {
        "spike_id": spike_packet["spike_id"],
        "spike_type": spike_packet["spike_type"],
        "primary_state": spike_packet["primary_state"],
        "secondary_states": spike_packet["secondary_states"],
        "topic": spike_packet["topic"],
        "priority": spike_packet["priority"],
        "blocker_type": spike_packet["blocker_type"],
        "action_policy": spike_packet["action_policy"],
        "completion_target": spike_packet["completion_target"],
        "trigger_reason": spike_packet["trigger_reason"],
        "evidence_notes": spike_packet["evidence_notes"],
    }
    system = (
        "你是 LIF-Memory 的语义审查器，不是控制器。"
        "你只能审查 topic、primary_state、secondary_states、completion_target 是否贴合证据。"
        "不要决定电压、阈值、冷却或最终 action_policy。"
        "必须只输出 JSON object。"
    )
    user = {
        "task": "Review this LIF-Memory spike. Return corrections only if needed.",
        "allowed_states": allowed_states,
        "allowed_output_schema": {
            "spike_id": "string",
            "is_correct": "boolean",
            "corrected_topic": "string|null",
            "corrected_primary_state": "string|null",
            "corrected_secondary_states": "array|null",
            "better_completion_target": "string|null",
            "risk": "string|null",
            "reason": "string",
        },
        "spike": compact_packet,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
    ]


def call_chat_completions(config: LLMConfig, messages: list[dict[str, str]]) -> str:
    url = f"{config.base_url}/chat/completions"
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

    data = json.loads(raw)
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"LLM response missing choices: {raw}")
    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"LLM response missing message content: {raw}")
    return content


def normalize_review(review: dict[str, object], spike_packet: dict[str, object]) -> dict[str, object]:
    normalized = dict(review)
    same_topic = normalized.get("corrected_topic") in (None, "", spike_packet.get("topic"))
    same_primary = normalized.get("corrected_primary_state") in (None, "", spike_packet.get("primary_state"))
    corrected_secondaries = normalized.get("corrected_secondary_states")
    if corrected_secondaries in (None, ""):
        same_secondary = True
    elif isinstance(corrected_secondaries, list):
        same_secondary = set(str(item) for item in corrected_secondaries) == set(
            str(item) for item in spike_packet.get("secondary_states", [])
        )
    else:
        same_secondary = False
    same_target = normalized.get("better_completion_target") in (
        None,
        "",
        spike_packet.get("completion_target"),
    )
    no_actual_correction = same_topic and same_primary and same_secondary and same_target

    if same_topic:
        normalized["corrected_topic"] = None
    if same_primary:
        normalized["corrected_primary_state"] = None
    if same_secondary:
        normalized["corrected_secondary_states"] = None
    if same_target:
        normalized["better_completion_target"] = None

    reason = str(normalized.get("reason", ""))
    says_no_correction = any(token in reason for token in ["无需修正", "无需更正", "不需要修正", "no correction"])
    if normalized.get("is_correct") is False and no_actual_correction and says_no_correction:
        normalized["is_correct"] = True
        normalized["normalization_note"] = "LLM returned false but proposed no actual correction."
    return normalized


def review_spike(
    spike_packet: dict[str, object],
    allowed_states: list[str],
    config: LLMConfig,
) -> dict[str, object]:
    try:
        content = call_chat_completions(config, review_prompt(spike_packet, allowed_states))
        review = normalize_review(extract_json_object(content), spike_packet)
        review.setdefault("spike_id", spike_packet.get("spike_id", ""))
        return review
    except Exception as exc:
        return {
            "spike_id": spike_packet.get("spike_id", ""),
            "is_correct": None,
            "error": str(exc),
            "reason": "LLM review failed; LIF decision was not changed.",
        }
