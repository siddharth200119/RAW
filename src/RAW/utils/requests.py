from typing import Optional, Dict, Any, Generator, AsyncGenerator, Union
import httpx
from .logger import Logger

class RequestsClient:
    def __init__(
        self,
        base_url: str = "",
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        auth: Optional[Any] = None,
        timeout: Optional[float] = 30.0,
        logger: Optional[Logger] = None,
    ):
        """
        Initialize the RequestsClient.

        Args:
            base_url: The base URL for the client.
            headers: Default headers to include in all requests.
            cookies: Default cookies to include in all requests.
            auth: Authentication credentials (e.g., tuple or httpx.Auth).
            timeout: Timeout in seconds for requests (default 30.0).
            logger: An instance of Logger from src/RAW/utils/logger.py.
        """
        self.logger = logger
        self.base_url = base_url
        self.headers = headers
        self.cookies = cookies
        self.auth = auth
        self.timeout = timeout

        self.client = httpx.Client(
            base_url=base_url,
            headers=headers,
            cookies=cookies,
            auth=auth,
            timeout=timeout
        )
        self.async_client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            cookies=cookies,
            auth=auth,
            timeout=timeout
        )

    def _log_request(self, method: str, url: str, **kwargs):
        if self.logger:
            self.logger.info(f"Making {method} request to: {url}")
            debug_info = {
                "headers": kwargs.get("headers", self.headers),
                "params": kwargs.get("params"),
                "json": kwargs.get("json"),
                "data": kwargs.get("data"),
            }
            self.logger.debug(f"Request Details: {debug_info}")

    def _log_response(self, response: httpx.Response, stream: bool = False):
        if self.logger:
            log_msg = f"Response Status Code: {response.status_code} | URL: {response.url}"
            if response.status_code >= 400:
                self.logger.error(log_msg)
            else:
                self.logger.info(log_msg)
            if not stream:
                try:
                    # Try to log JSON response if possible, otherwise text
                    if "application/json" in response.headers.get("Content-Type", ""):
                        self.logger.debug(f"Response Body: {response.text}") 
                    else:
                        self.logger.debug(f"Response Body (text): {response.text}")
                except Exception:
                    self.logger.debug("Response Body: <Could not decode>")
            else:
                 self.logger.debug("Response Body: <Streamed Content>")

    def request_sync(
        self,
        method: str,
        url: str,
        stream: bool = False,
        raw_response: bool = False,
        **kwargs
    ) -> Union[httpx.Response, Generator[bytes, None, None]]:
        """
        Make a synchronous request.

        If stream=False, returns the httpx.Response object.
        If stream=True and raw_response=False, returns a generator yielding bytes chunks.
        If stream=True and raw_response=True, returns the raw httpx.Response object
          with the connection held open — the caller is responsible for iterating
          response.iter_bytes() and the response context.
        """
        self._log_request(method, url, **kwargs)

        if stream:
            ctx = self.client.stream(method, url, **kwargs)
            response = ctx.__enter__()
            self._log_response(response, stream=True)

            if raw_response:
                # Pin ctx on the response so it isn't GC'd while the caller holds the response.
                # Caller must call response.close() when done.
                response._stream_ctx = ctx
                return response

            def stream_generator():
                try:
                    for chunk in response.iter_bytes():
                        yield chunk
                finally:
                    ctx.__exit__(None, None, None)

            return stream_generator()

        else:
            response = self.client.request(method, url, **kwargs)
            self._log_response(response, stream=False)
            return response

    async def request_async(
        self,
        method: str,
        url: str,
        stream: bool = False,
        raw_response: bool = False,
        **kwargs
    ) -> Union[httpx.Response, AsyncGenerator[bytes, None]]:
        """
        Make an asynchronous request.

        If stream=False, returns the httpx.Response object.
        If stream=True and raw_response=False, returns an async generator yielding bytes chunks.
        If stream=True and raw_response=True, returns the raw httpx.Response object
          with the connection held open — the caller is responsible for iterating
          response.aiter_bytes() and the response context.
        """
        self._log_request(method, url, **kwargs)

        if stream:
            ctx = self.async_client.stream(method, url, **kwargs)
            response = await ctx.__aenter__()
            self._log_response(response, stream=True)

            if raw_response:
                # Pin ctx on the response so it isn't GC'd while the caller holds the response.
                # Caller must call await response.aclose() when done.
                response._stream_ctx = ctx
                return response

            async def stream_generator():
                content_buffer = b""
                try:
                    async for chunk in response.aiter_bytes():
                        content_buffer += chunk
                        yield chunk
                finally:
                    await ctx.__aexit__(None, None, None)
                    if self.logger:
                        try:
                            text = content_buffer.decode('utf-8')
                            self.logger.debug(f"Full Streamed Response: {text}")
                        except Exception:
                            self.logger.debug(f"Full Streamed Response: <Could not decode {len(content_buffer)} bytes>")

            return stream_generator()

        else:
            response = await self.async_client.request(method, url, **kwargs)
            self._log_response(response, stream=False)
            return response

    def close(self):
        self.client.close()

    async def aclose(self):
        await self.async_client.aclose()

    def get(self, url: str, stream: bool = False, raw_response: bool = False, is_async: bool = False, **kwargs):
        if is_async:
            return self.request_async("GET", url, stream=stream, raw_response=raw_response, **kwargs)
        return self.request_sync("GET", url, stream=stream, raw_response=raw_response, **kwargs)

    def post(self, url: str, stream: bool = False, raw_response: bool = False, is_async: bool = False, **kwargs):
        if is_async:
            return self.request_async("POST", url, stream=stream, raw_response=raw_response, **kwargs)
        return self.request_sync("POST", url, stream=stream, raw_response=raw_response, **kwargs)

    def put(self, url: str, stream: bool = False, raw_response: bool = False, is_async: bool = False, **kwargs):
        if is_async:
            return self.request_async("PUT", url, stream=stream, raw_response=raw_response, **kwargs)
        return self.request_sync("PUT", url, stream=stream, raw_response=raw_response, **kwargs)

    def patch(self, url: str, stream: bool = False, raw_response: bool = False, is_async: bool = False, **kwargs):
        if is_async:
            return self.request_async("PATCH", url, stream=stream, raw_response=raw_response, **kwargs)
        return self.request_sync("PATCH", url, stream=stream, raw_response=raw_response, **kwargs)

    def delete(self, url: str, stream: bool = False, raw_response: bool = False, is_async: bool = False, **kwargs):
        if is_async:
            return self.request_async("DELETE", url, stream=stream, raw_response=raw_response, **kwargs)
        return self.request_sync("DELETE", url, stream=stream, raw_response=raw_response, **kwargs)
