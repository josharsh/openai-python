"""Test retry logging functionality with retry count in INFO messages."""

import logging
from typing import Any
from unittest import mock

import httpx
import pytest

import openai
from openai import APIConnectionError


class TestRetryLogging:
    """Test that retry attempts are logged at INFO level with retry count information."""

    @pytest.fixture
    def mock_response_with_retry(self) -> httpx.Response:
        """Create a mock response that triggers retries."""
        return httpx.Response(
            status_code=429,  # Rate limit error to trigger retry
            headers={"retry-after": "0.1"},  # Small retry delay for tests
            json={"error": {"message": "Rate limit exceeded"}},
        )

    @pytest.fixture  
    def client_with_retries(self) -> openai.OpenAI:
        """Create a client configured with retry settings."""
        return openai.OpenAI(
            api_key="test-key",
            max_retries=3,
            base_url="http://localhost:8000",
        )

    def test_retry_logging_format_sync(self, client_with_retries: openai.OpenAI, caplog: Any) -> None:
        """Test that sync client logs retries with correct format including retry count."""
        with mock.patch.object(
            client_with_retries._client,
            "send",
            side_effect=[
                httpx.Response(429, json={"error": {"message": "Rate limit"}}, headers={"retry-after": "0.01"}),
                httpx.Response(429, json={"error": {"message": "Rate limit"}}, headers={"retry-after": "0.01"}),
                httpx.Response(200, json={"id": "test", "object": "test", "created": 1234567890}),
            ],
        ):
            # Set log level to INFO to capture retry messages
            caplog.set_level(logging.INFO)
            
            # Make a request that will succeed after 2 retries
            response = client_with_retries.models.list()
            assert response is not None
            
            # Check that INFO logs contain retry count information
            info_logs = [record for record in caplog.records if record.levelname == "INFO"]
            
            # Should have 2 retry log messages (not 3, as first attempt isn't a retry)
            retry_logs = [log for log in info_logs if "Retrying request" in log.message]
            assert len(retry_logs) == 2
            
            # Check format of retry messages
            assert "(retry 1 of 3)" in retry_logs[0].message
            assert "(retry 2 of 3)" in retry_logs[1].message
            
            # Ensure no DEBUG logs about retries remaining
            debug_logs = [record for record in caplog.records if record.levelname == "DEBUG"]
            retry_debug_logs = [log for log in debug_logs if "retries left" in log.message or "retry left" in log.message]
            assert len(retry_debug_logs) == 0

    @pytest.mark.asyncio
    async def test_retry_logging_format_async(self, caplog: Any) -> None:
        """Test that async client logs retries with correct format including retry count."""
        client = openai.AsyncOpenAI(
            api_key="test-key",
            max_retries=2,
            base_url="http://localhost:8000",
        )
        
        with mock.patch.object(
            client._client,
            "send",
            side_effect=[
                httpx.Response(500, json={"error": {"message": "Server error"}}, headers={}),
                httpx.Response(200, json={"id": "test", "object": "test", "created": 1234567890}),
            ],
        ):
            # Set log level to INFO
            caplog.set_level(logging.INFO)
            
            # Make a request that will succeed after 1 retry
            response = await client.models.list()
            assert response is not None
            
            # Check retry log format
            info_logs = [record for record in caplog.records if record.levelname == "INFO"]
            retry_logs = [log for log in info_logs if "Retrying request" in log.message]
            
            assert len(retry_logs) == 1
            assert "(retry 1 of 2)" in retry_logs[0].message

    def test_retry_count_progression(self, client_with_retries: openai.OpenAI, caplog: Any) -> None:
        """Test that retry count increments correctly across multiple retries."""
        # Force all retries to fail to test max retry scenario
        with mock.patch.object(
            client_with_retries._client,
            "send",
            side_effect=httpx.ConnectTimeout("Connection timeout"),
        ):
            caplog.set_level(logging.INFO)
            
            # This should fail after exhausting all retries
            with pytest.raises(APIConnectionError):
                client_with_retries.models.list()
            
            # Check all retry messages
            info_logs = [record for record in caplog.records if record.levelname == "INFO"]
            retry_logs = [log for log in info_logs if "Retrying request" in log.message]
            
            # With max_retries=3, we should see 3 retry attempts
            assert len(retry_logs) == 3
            assert "(retry 1 of 3)" in retry_logs[0].message
            assert "(retry 2 of 3)" in retry_logs[1].message  
            assert "(retry 3 of 3)" in retry_logs[2].message

    def test_single_retry_logging(self, caplog: Any) -> None:
        """Test retry logging when max_retries is 1."""
        client = openai.OpenAI(
            api_key="test-key",
            max_retries=1,
            base_url="http://localhost:8000",
        )
        
        with mock.patch.object(
            client._client,
            "send", 
            side_effect=[
                httpx.Response(503, json={"error": {"message": "Service unavailable"}}),
                httpx.Response(200, json={"id": "test", "object": "test", "created": 1234567890}),
            ],
        ):
            caplog.set_level(logging.INFO)
            
            response = client.models.list()
            assert response is not None
            
            retry_logs = [r for r in caplog.records if r.levelname == "INFO" and "Retrying request" in r.message]
            assert len(retry_logs) == 1
            assert "(retry 1 of 1)" in retry_logs[0].message

    def test_no_retries_no_logs(self, caplog: Any) -> None:
        """Test that successful requests without retries don't log retry messages."""
        client = openai.OpenAI(
            api_key="test-key",
            max_retries=3,
            base_url="http://localhost:8000",
        )
        
        with mock.patch.object(
            client._client,
            "send",
            return_value=httpx.Response(200, json={"id": "test", "object": "test", "created": 1234567890}),
        ):
            caplog.set_level(logging.INFO)
            
            response = client.models.list()
            assert response is not None
            
            # No retry logs should be present
            retry_logs = [r for r in caplog.records if "Retrying request" in r.message]
            assert len(retry_logs) == 0

    def test_retry_url_logged_correctly(self, client_with_retries: openai.OpenAI, caplog: Any) -> None:
        """Test that the correct URL is logged in retry messages."""
        with mock.patch.object(
            client_with_retries._client,
            "send",
            side_effect=[
                httpx.Response(408, json={"error": {"message": "Timeout"}}),
                httpx.Response(200, json={"id": "test", "object": "test", "created": 1234567890}),
            ],
        ):
            caplog.set_level(logging.INFO)
            
            client_with_retries.models.list()
            
            retry_logs = [r for r in caplog.records if r.levelname == "INFO" and "Retrying request" in r.message]
            assert len(retry_logs) == 1
            
            # Check that URL is included in the log message
            assert "/models" in retry_logs[0].message or "models" in retry_logs[0].message