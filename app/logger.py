import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from config.settings import settings

logger = logging.getLogger(__name__)


def _ensure_log_directory() -> None:
    log_dir = os.path.dirname(settings.log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)


def _write_log_entry_sync(log_entry: dict) -> None:
    _ensure_log_directory()
    try:
        with open(settings.log_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.error("Failed to write route log: %s", exc)


async def log_route_decision(
    *,
    intent: str,
    confidence: float,
    user_message: str,
    final_response: str,
    routing_method: str = "auto",
    model_used: str | None = None,
    latency_ms: float | None = None
) -> dict:
    _ensure_log_directory()
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "intent": intent,
        "confidence": confidence,
        "user_message": user_message,
        "final_response": final_response,
        "routing_method": routing_method,
        "model_used": model_used or settings.openai_model,
        "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
    }
    await asyncio.to_thread(_write_log_entry_sync, log_entry)
    logger.debug(
        "Route decision logged asynchronously: intent=%s, confidence=%.2f",
        intent,
        confidence,
    )
    return log_entry
