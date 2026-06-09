import httpx
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
from RAW.utils.requests import RequestsClient
from RAW.utils.logger import Logger


def test_requests_client_init():
    client = RequestsClient(base_url="https://api.example.com", timeout=10.0)
    assert client.base_url == "https://api.example.com"
    assert client.timeout == 10.0
    assert isinstance(client.client, httpx.Client)
    assert isinstance(client.async_client, httpx.AsyncClient)
    client.close()


def test_requests_client_sync_methods():
    client = RequestsClient(base_url="https://api.example.com")
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.url = "https://api.example.com/test"
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.text = '{"success": true}'

    with patch.object(client.client, "request", return_value=mock_response) as mock_request:
        # Test GET
        res = client.get("/test")
        mock_request.assert_called_once_with("GET", "/test")
        assert res.status_code == 200

        # Test POST
        mock_request.reset_mock()
        res = client.post("/test", json={"data": 123})
        mock_request.assert_called_once_with("POST", "/test", json={"data": 123})

        # Test PUT, PATCH, DELETE
        for method in ["PUT", "PATCH", "DELETE"]:
            mock_request.reset_mock()
            getattr(client, method.lower())("/test")
            mock_request.assert_called_once_with(method, "/test")

    client.close()


@pytest.mark.asyncio
async def test_requests_client_async_methods():
    client = RequestsClient(base_url="https://api.example.com")
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 201
    mock_response.url = "https://api.example.com/test"
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.text = '{"created": true}'

    with patch.object(client.async_client, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response

        # Test GET async
        res = await client.get("/test", is_async=True)
        mock_request.assert_called_once_with("GET", "/test")
        assert res.status_code == 201

        # Test POST async
        mock_request.reset_mock()
        res = await client.post("/test", is_async=True, json={"data": 456})
        mock_request.assert_called_once_with("POST", "/test", json={"data": 456})

        # Test PUT, PATCH, DELETE async
        for method in ["PUT", "PATCH", "DELETE"]:
            mock_request.reset_mock()
            await getattr(client, method.lower())("/test", is_async=True)
            mock_request.assert_called_once_with(method, "/test")

    await client.aclose()


def test_requests_client_sync_stream():
    client = RequestsClient(base_url="https://api.example.com")

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.url = "https://api.example.com/stream"
    mock_response.iter_bytes.return_value = [b"chunk1", b"chunk2"]

    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_response

    with patch.object(client.client, "stream", return_value=mock_ctx) as mock_stream:
        generator = client.get("/stream", stream=True)
        chunks = list(generator)

        mock_stream.assert_called_once_with("GET", "/stream")
        assert chunks == [b"chunk1", b"chunk2"]
        mock_ctx.__exit__.assert_called_once()

    client.close()


@pytest.mark.asyncio
async def test_requests_client_async_stream():
    client = RequestsClient(base_url="https://api.example.com")

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.url = "https://api.example.com/stream"

    async def mock_aiter_bytes():
        yield b"async_chunk1"
        yield b"async_chunk2"

    mock_response.aiter_bytes = mock_aiter_bytes

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)

    with patch.object(client.async_client, "stream", return_value=mock_ctx) as mock_stream:
        generator = await client.get("/stream", stream=True, is_async=True)
        chunks = []
        async for chunk in generator:
            chunks.append(chunk)

        mock_stream.assert_called_once_with("GET", "/stream")
        assert chunks == [b"async_chunk1", b"async_chunk2"]
        mock_ctx.__aexit__.assert_called_once()

    await client.aclose()


def test_requests_client_logging():
    mock_logger = MagicMock(spec=Logger)
    client = RequestsClient(base_url="https://api.example.com", logger=mock_logger)

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.url = "https://api.example.com/log-test"
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.text = '{"ok": true}'

    with patch.object(client.client, "request", return_value=mock_response):
        client.get("/log-test")
        assert mock_logger.info.call_count >= 2
        assert mock_logger.debug.call_count >= 1

    client.close()
