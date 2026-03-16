import logging
import time
from app.exceptions import RoutingError
from app.models import IntentResult
from app.services.llm_service import llm_service
from config.prompts import CLARIFICATION_PROMPT, EXPERT_PROMPTS, Intent
from config.settings import settings

logger = logging.getLogger(__name__)


def _should_clarify(intent_result: IntentResult) -> bool:
    if intent_result.intent == Intent.UNCLEAR:
        return True
    if intent_result.confidence < settings.confidence_threshold:
        logger.info(
            "Confidence %.2f below threshold %.2f — requesting clarification.",
            intent_result.confidence,
            settings.confidence_threshold,
        )
        return True
    return False


async def route_and_respond(message: str, intent_result: IntentResult) -> str:
    if _should_clarify(intent_result):
        logger.info(
            "Returning clarification prompt for intent=%s.", intent_result.intent.value
        )
        return CLARIFICATION_PROMPT
    expert_prompt = EXPERT_PROMPTS.get(intent_result.intent)
    if expert_prompt is None:
        logger.error(
            "No expert prompt found for intent=%s.", intent_result.intent.value
        )
        raise RoutingError(
            f"No expert persona configured for intent '{intent_result.intent.value}'."
        )
    start_time = time.perf_counter()
    try:
        response_text = await llm_service.get_chat_completion(
            system_prompt=expert_prompt,
            user_message=message,
            temperature=0.7,
            max_tokens=512
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info("Expert response completed in %.1fms", elapsed_ms)
        return response_text
    except Exception as exc:
        logger.exception("Error during expert response generation.")
        raise RoutingError(f"Expert response generation failed: {exc}") from exc
