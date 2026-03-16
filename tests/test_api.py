import pytest
from unittest.mock import patch, AsyncMock
from tests.conftest import make_mock_completion

TEST_MESSAGES = [
    ("how do I sort a list in python", "code", "Clear code question"),
    ("write a recursive fibonacci function", "code", "Code: algorithm question"),
    (
        "what is the difference between a stack and a queue",
        "code",
        "Code: data structures",
    ),
    ("explain pivot table", "data", "Clear data question"),
    ("how to calculate standard deviation in SQL", "data", "Data: SQL statistics"),
    (
        "this paragraph sounds awkward, can you improve it",
        "writing",
        "Clear writing question",
    ),
    ("proofread my email for grammar mistakes", "writing", "Writing: proofreading"),
    ("career advice for AI engineer", "career", "Clear career question"),
    (
        "how to prepare for a system design interview",
        "career",
        "Career: interview prep",
    ),
    ("I need help with my data science resume", None, "Ambiguous: career + data"),
    ("help me write a python script for data analysis", None, "Ambiguous: code + data"),
    ("fxi thsi bug pls", None, "Typo: likely code"),
    ("halp me wite better", None, "Typo: likely writing"),
    ("hello", "unclear", "Greeting"),
    ("what's the weather like today", "unclear", "Off-topic"),
    ("@code fix this bug in my code", "code", "Manual override: @code"),
    ("@data analyze this sales dataset", "data", "Manual override: @data"),
]


@pytest.mark.asyncio
class TestRouteEndpoint:

    @pytest.mark.parametrize(
        "message,expected_intent,description",
        [msg for msg in TEST_MESSAGES if msg[1] is not None],
    )
    async def test_clear_intent_messages(
        self, client, message, expected_intent, description
    ):
        from app.models import IntentResult
        from config.prompts import Intent
        intent_enum = Intent(expected_intent) if expected_intent != "unclear" else Intent.UNCLEAR
        with patch("app.main.classify_intent", new_callable=AsyncMock, return_value=IntentResult(intent=intent_enum, confidence=0.92)):
            with patch("app.main.route_and_respond", new_callable=AsyncMock, return_value="Expert response for your question."):
                response = await client.post("/route", json={"message": message})
        assert response.status_code == 200, f"Failed: {description}"
        data = response.json()
        assert "intent" in data
        assert "response" in data
        assert (
            data["intent"]["intent"] == expected_intent
        ), f"Expected '{expected_intent}' for: {description}"

    async def test_empty_message_rejected(self, client):
        response = await client.post("/route", json={"message": ""})
        assert response.status_code == 422

    async def test_whitespace_only_message(self, client):
        response = await client.post("/route", json={"message": "   "})
        assert response.status_code == 422

    async def test_missing_message_field(self, client):
        response = await client.post("/route", json={"text": "hello"})
        assert response.status_code == 422

    async def test_response_structure(self, client):
        from app.models import IntentResult
        from config.prompts import Intent
        with patch("app.main.classify_intent", new_callable=AsyncMock, return_value=IntentResult(intent=Intent.CODE, confidence=0.95)):
            with patch("app.main.route_and_respond", new_callable=AsyncMock, return_value="Use list.sort() method."):
                response = await client.post(
                    "/route", json={"message": "how to sort in python"}
                )
        data = response.json()
        assert "intent" in data
        assert "response" in data
        assert "intent" in data["intent"]
        assert "confidence" in data["intent"]
        assert isinstance(data["intent"]["confidence"], float)
        assert isinstance(data["response"], str)

    async def test_health_endpoint(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "llm-prompt-router"

    async def test_unclear_returns_clarification(self, client):
        from app.models import IntentResult
        from config.prompts import Intent
        with patch("app.main.classify_intent", new_callable=AsyncMock, return_value=IntentResult(intent=Intent.UNCLEAR, confidence=0.2)):
            with patch("app.main.route_and_respond", new_callable=AsyncMock, return_value="I'd love to help, but I want to make sure I connect you with the right expert. Could you provide a bit more detail about what you need?"):
                response = await client.post("/route", json={"message": "hello"})
        assert response.status_code == 200
        data = response.json()
        assert data["intent"]["intent"] == "unclear"
        assert (
            "more detail" in data["response"].lower()
            or "rephrase" in data["response"].lower()
        )

    async def test_low_confidence_returns_clarification(self, client):
        from app.models import IntentResult
        from config.prompts import Intent
        with patch("app.main.classify_intent", new_callable=AsyncMock, return_value=IntentResult(intent=Intent.CODE, confidence=0.3)):
            with patch("app.main.route_and_respond", new_callable=AsyncMock, return_value="I'd love to help, but I want to make sure I connect you with the right expert. Could you provide a bit more detail about what you need?"):
                response = await client.post("/route", json={"message": "help"})
        assert response.status_code == 200
        data = response.json()
        assert (
            "rephrase" in data["response"].lower()
            or "more detail" in data["response"].lower()
        )

    async def test_manual_override_bypasses_classifier(self, client):
        from app.models import IntentResult
        from config.prompts import Intent
        with patch("app.main.classify_intent", new_callable=AsyncMock, return_value=IntentResult(intent=Intent.CODE, confidence=1.0)):
            with patch("app.main.route_and_respond", new_callable=AsyncMock, return_value="Here's the fix for your bug."):
                response = await client.post(
                    "/route", json={"message": "@code fix this bug"}
                )
        assert response.status_code == 200
        data = response.json()
        assert data["intent"]["intent"] == "code"
        assert data["intent"]["confidence"] == 1.0
