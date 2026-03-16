import os
import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-for-testing-only")
os.environ.setdefault("OPENAI_MODEL", "gpt-4o-mini")
os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.7")
os.environ.setdefault("LOG_FILE_PATH", "logs/test_route_log.jsonl")
from httpx import ASGITransport, AsyncClient
from app.main import app
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def make_mock_completion(content: str):
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response
