import json
import logging
import re
import time
from app.exceptions import ClassificationError
from app.models import IntentResult
from app.services.llm_service import llm_service
from config.prompts import CLASSIFICATION_SYSTEM_PROMPT, Intent

logger = logging.getLogger(__name__)
MANUAL_OVERRIDE_PREFIXES: dict[str, Intent] = {
    "@code": Intent.CODE,
    "@data": Intent.DATA,
    "@writing": Intent.WRITING,
    "@career": Intent.CAREER,
}


def _detect_manual_override(message: str) -> IntentResult | None:
    lowered = message.strip().lower()
    for prefix, intent in MANUAL_OVERRIDE_PREFIXES.items():
        if lowered.startswith(prefix):
            logger.info("Manual override detected: %s → %s", prefix, intent.value)
            return IntentResult(intent=intent, confidence=1.0)
    return None


def strip_override_prefix(message: str) -> str:
    lowered = message.strip().lower()
    for prefix in MANUAL_OVERRIDE_PREFIXES:
        if lowered.startswith(prefix):
            return message.strip()[len(prefix) :].strip()
    return message


def _parse_llm_response(raw: str) -> IntentResult:
    try:
        data = json.loads(raw.strip())
        return _validate_parsed_data(data)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Level 1 JSON parse failed, attempting regex extraction.")
    json_pattern = (
        '\\{[^{}]*"intent"\\s*:\\s*"[^"]*"[^{}]*"confidence"\\s*:\\s*[\\d.]+[^{}]*\\}'
    )
    match = re.search(json_pattern, raw, re.IGNORECASE)
    if match:
        try:
            data = json.loads(match.group())
            return _validate_parsed_data(data)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Level 2 regex JSON parse failed.")
    logger.error("All JSON parsing attempts failed. Raw LLM output: %s", raw[:200])
    return IntentResult(intent=Intent.UNCLEAR, confidence=0.0)


def _validate_parsed_data(data: dict) -> IntentResult:
    intent_str = str(data.get("intent", "")).lower().strip()
    confidence_raw = data.get("confidence", 0.0)
    try:
        intent = Intent(intent_str)
    except ValueError:
        logger.warning("Unknown intent '%s', falling back to unclear.", intent_str)
        intent = Intent.UNCLEAR
    try:
        confidence = float(confidence_raw)
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.0
    return IntentResult(intent=intent, confidence=confidence)


async def classify_intent(message: str) -> IntentResult:
    override = _detect_manual_override(message)
    if override is not None:
        return override
    start_time = time.perf_counter()
    try:
        raw_content = await llm_service.get_chat_completion(
            system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
            user_message=message,
            temperature=0.1,
            max_tokens=100
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Classification completed in %.1fms. Raw: %s", elapsed_ms, raw_content[:100]
        )
        return _parse_llm_response(raw_content)
    except Exception as exc:
        logger.exception("Unexpected error during classification.")
        raise ClassificationError(f"Unexpected classification failure: {exc}") from exc
