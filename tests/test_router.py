import pytest
from unittest.mock import patch, AsyncMock
from app.models import IntentResult
from app.router import route_and_respond, _should_clarify
from config.prompts import CLARIFICATION_PROMPT, Intent
from tests.conftest import make_mock_completion


class TestShouldClarify:

    def test_unclear_always_clarifies(self):
        result = IntentResult(intent=Intent.UNCLEAR, confidence=0.5)
        assert _should_clarify(result) is True

    def test_low_confidence_clarifies(self):
        result = IntentResult(intent=Intent.CODE, confidence=0.3)
        assert _should_clarify(result) is True

    def test_high_confidence_does_not_clarify(self):
        result = IntentResult(intent=Intent.CODE, confidence=0.9)
        assert _should_clarify(result) is False

    def test_threshold_boundary(self):
        result = IntentResult(intent=Intent.DATA, confidence=0.7)
        assert _should_clarify(result) is False


class TestRouteAndRespond:

    @pytest.mark.asyncio
    async def test_routes_to_code_expert(self):
        intent = IntentResult(intent=Intent.CODE, confidence=0.95)
        with patch("app.services.llm_service.llm_service.get_chat_completion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Here's how to sort a list in Python..."
            response = await route_and_respond("how to sort", intent)
            assert "sort" in response.lower()
            mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_to_data_expert(self):
        intent = IntentResult(intent=Intent.DATA, confidence=0.88)
        with patch("app.services.llm_service.llm_service.get_chat_completion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "A pivot table summarizes data by grouping..."
            response = await route_and_respond("explain pivot table", intent)
            assert "pivot" in response.lower()

    @pytest.mark.asyncio
    async def test_routes_to_writing_expert(self):
        intent = IntentResult(intent=Intent.WRITING, confidence=0.9)
        with patch("app.services.llm_service.llm_service.get_chat_completion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "The sentence could be improved by..."
            response = await route_and_respond("this paragraph sounds awkward", intent)
            assert "improved" in response.lower()

    @pytest.mark.asyncio
    async def test_routes_to_career_expert(self):
        intent = IntentResult(intent=Intent.CAREER, confidence=0.85)
        with patch("app.services.llm_service.llm_service.get_chat_completion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "For an AI engineering career, focus on..."
            response = await route_and_respond("career advice for AI engineer", intent)
            assert "career" in response.lower()

    @pytest.mark.asyncio
    async def test_unclear_returns_clarification(self):
        intent = IntentResult(intent=Intent.UNCLEAR, confidence=0.3)
        response = await route_and_respond("hello", intent)
        assert response == CLARIFICATION_PROMPT

    @pytest.mark.asyncio
    async def test_low_confidence_returns_clarification(self):
        intent = IntentResult(intent=Intent.CODE, confidence=0.4)
        response = await route_and_respond("something vague", intent)
        assert response == CLARIFICATION_PROMPT
