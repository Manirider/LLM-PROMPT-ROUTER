import pytest
from unittest.mock import AsyncMock, patch
from app.classifier import (
    _detect_manual_override,
    _parse_llm_response,
    classify_intent,
    strip_override_prefix,
)
from app.models import IntentResult
from config.prompts import Intent
from tests.conftest import make_mock_completion


class TestParseLLMResponse:

    def test_valid_json(self):
        result = _parse_llm_response('{"intent": "code", "confidence": 0.95}')
        assert result.intent == Intent.CODE
        assert result.confidence == 0.95

    def test_json_with_whitespace(self):
        result = _parse_llm_response('  {"intent": "data", "confidence": 0.8}  ')
        assert result.intent == Intent.DATA
        assert result.confidence == 0.8

    def test_json_in_markdown_backticks(self):
        raw = '```json\n{"intent": "writing", "confidence": 0.9}\n```'
        result = _parse_llm_response(raw)
        assert result.intent == Intent.WRITING
        assert result.confidence == 0.9

    def test_json_with_surrounding_text(self):
        raw = 'Based on analysis, the result is {"intent": "career", "confidence": 0.85} as expected.'
        result = _parse_llm_response(raw)
        assert result.intent == Intent.CAREER
        assert result.confidence == 0.85

    def test_complete_garbage_fallback(self):
        result = _parse_llm_response("I cannot determine the intent of this message.")
        assert result.intent == Intent.UNCLEAR
        assert result.confidence == 0.0

    def test_empty_string_fallback(self):
        result = _parse_llm_response("")
        assert result.intent == Intent.UNCLEAR
        assert result.confidence == 0.0

    def test_unknown_intent_falls_back(self):
        result = _parse_llm_response('{"intent": "cooking", "confidence": 0.9}')
        assert result.intent == Intent.UNCLEAR

    def test_confidence_clamped_above_one(self):
        result = _parse_llm_response('{"intent": "code", "confidence": 1.5}')
        assert result.confidence == 1.0

    def test_confidence_clamped_below_zero(self):
        result = _parse_llm_response('{"intent": "code", "confidence": -0.5}')
        assert result.confidence == 0.0

    def test_non_numeric_confidence(self):
        result = _parse_llm_response('{"intent": "code", "confidence": "high"}')
        assert result.confidence == 0.0


class TestManualOverride:

    def test_code_override(self):
        result = _detect_manual_override("@code fix this function")
        assert result is not None
        assert result.intent == Intent.CODE
        assert result.confidence == 1.0

    def test_data_override(self):
        result = _detect_manual_override("@data analyze this CSV")
        assert result is not None
        assert result.intent == Intent.DATA

    def test_writing_override(self):
        result = _detect_manual_override("@writing improve this paragraph")
        assert result is not None
        assert result.intent == Intent.WRITING

    def test_career_override(self):
        result = _detect_manual_override("@career review my resume")
        assert result is not None
        assert result.intent == Intent.CAREER

    def test_no_override(self):
        result = _detect_manual_override("how do I sort a list")
        assert result is None

    def test_case_insensitive(self):
        result = _detect_manual_override("@CODE fix this bug")
        assert result is not None
        assert result.intent == Intent.CODE


class TestStripOverridePrefix:

    def test_strip_code_prefix(self):
        result = strip_override_prefix("@code fix this function")
        assert result == "fix this function"

    def test_strip_data_prefix(self):
        result = strip_override_prefix("@data analyze sales data")
        assert result == "analyze sales data"

    def test_no_prefix_unchanged(self):
        result = strip_override_prefix("how do I sort a list")
        assert result == "how do I sort a list"


class TestClassifyIntent:

    @pytest.mark.asyncio
    async def test_classify_with_valid_response(self):
        with patch("app.services.llm_service.llm_service.get_chat_completion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = '{"intent": "code", "confidence": 0.92}'
            result = await classify_intent("how do I sort a list in python")
            assert result.intent == Intent.CODE
            assert result.confidence == 0.92

    @pytest.mark.asyncio
    async def test_classify_with_manual_override(self):
        result = await classify_intent("@data analyze this dataset")
        assert result.intent == Intent.DATA
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_classify_with_malformed_json(self):
        with patch("app.services.llm_service.llm_service.get_chat_completion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "I'm not sure what to classify this as."
            result = await classify_intent("asdfghjkl")
            assert result.intent == Intent.UNCLEAR
            assert result.confidence == 0.0
