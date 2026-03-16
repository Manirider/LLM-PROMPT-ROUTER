import json
import os
import tempfile
import pytest
from unittest.mock import patch
from app.logger import log_route_decision


class TestLogRouteDecision:

    @pytest.mark.asyncio
    async def test_creates_log_entry(self, tmp_path):
        log_file = str(tmp_path / "test_log.jsonl")
        with patch("app.logger.settings") as mock_settings:
            mock_settings.log_file_path = log_file
            mock_settings.openai_model = "gpt-4o-mini"
            entry = await log_route_decision(
                intent="code",
                confidence=0.95,
                user_message="how to sort a list",
                final_response="Use sorted() or list.sort()...",
                routing_method="auto",
                latency_ms=150.5,
            )
        assert entry["intent"] == "code"
        assert entry["confidence"] == 0.95
        assert entry["user_message"] == "how to sort a list"
        assert entry["final_response"] == "Use sorted() or list.sort()..."
        assert entry["routing_method"] == "auto"
        assert entry["latency_ms"] == 150.5
        assert "timestamp" in entry
        assert os.path.exists(log_file)
        with open(log_file, "r", encoding="utf-8") as f:
            line = f.readline()
            data = json.loads(line)
            assert data["intent"] == "code"

    @pytest.mark.asyncio
    async def test_appends_multiple_entries(self, tmp_path):
        log_file = str(tmp_path / "multi_log.jsonl")
        with patch("app.logger.settings") as mock_settings:
            mock_settings.log_file_path = log_file
            mock_settings.openai_model = "gpt-4o-mini"
            await log_route_decision(
                intent="code",
                confidence=0.9,
                user_message="msg1",
                final_response="resp1",
            )
            await log_route_decision(
                intent="data",
                confidence=0.85,
                user_message="msg2",
                final_response="resp2",
            )
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 2
            assert json.loads(lines[0])["intent"] == "code"
            assert json.loads(lines[1])["intent"] == "data"

    @pytest.mark.asyncio
    async def test_creates_directory_if_missing(self, tmp_path):
        log_file = str(tmp_path / "nested" / "dir" / "log.jsonl")
        with patch("app.logger.settings") as mock_settings:
            mock_settings.log_file_path = log_file
            mock_settings.openai_model = "gpt-4o-mini"
            await log_route_decision(
                intent="writing",
                confidence=0.8,
                user_message="improve this",
                final_response="Here's a suggestion...",
            )
        assert os.path.exists(log_file)

    @pytest.mark.asyncio
    async def test_timestamp_is_iso_format(self, tmp_path):
        log_file = str(tmp_path / "time_log.jsonl")
        with patch("app.logger.settings") as mock_settings:
            mock_settings.log_file_path = log_file
            mock_settings.openai_model = "gpt-4o-mini"
            entry = await log_route_decision(
                intent="career",
                confidence=0.75,
                user_message="resume help",
                final_response="Sure, let's review...",
            )
        assert "T" in entry["timestamp"]

    @pytest.mark.asyncio
    async def test_default_model_used(self, tmp_path):
        log_file = str(tmp_path / "model_log.jsonl")
        with patch("app.logger.settings") as mock_settings:
            mock_settings.log_file_path = log_file
            mock_settings.openai_model = "gpt-4o-mini"
            entry = await log_route_decision(
                intent="code",
                confidence=0.9,
                user_message="test",
                final_response="response",
            )
        assert entry["model_used"] == "gpt-4o-mini"
