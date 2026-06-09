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
        **kwargs
    ) -> Union[httpx.Response, Generator[bytes, None, None]]:
        """
        Make a synchronous request.

        If stream=True, returns a generator yielding bytes.
        If stream=False, returns the httpx.Response object.
        """
        # If url starts with http, httpx handles ignoring base_url automatically if it's an absolute URL.
        # However, checking explicitly to be safe or if custom logic is needed, but httpx default behavior matches requirement.
        
        self._log_request(method, url, **kwargs)

        if stream:
            # For streaming, we need to use the client.stream() context manager manually or return result.
            # But the user asked for a generator.
            # We cannot easily yield from a context manager that closes immediately.
            # We typically keep the connection open. verify httpx pattern.
            # httpx.stream() is a context manager.
            
            # Implementation for generator using stream context:
            # We'll rely on the client.stream context manager but we need to yield out of it.
            # Actually, standard usage is `with client.stream(...) as response:`.
            # If we want to return a generator that the user iterates over, we might need to handle the context differently.
            # Let's encapsulate the stream generator.
            
            ctx = self.client.stream(method, url, **kwargs)
            response = ctx.__enter__()
            self._log_response(response, stream=True)
            
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
        **kwargs
    ) -> Union[httpx.Response, AsyncGenerator[bytes, None]]:
        """
        Make an asynchronous request.

        If stream=True, returns an async generator yielding bytes.
        If stream=False, returns the httpx.Response object.
        """
        self._log_request(method, url, **kwargs)

        if stream:
            ctx = self.async_client.stream(method, url, **kwargs)
            response = await ctx.__aenter__()
            self._log_response(response, stream=True)
            
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

    def get(self, url: str, stream: bool = False, is_async: bool = False, **kwargs):
        if is_async:
            return self.request_async("GET", url, stream=stream, **kwargs)
        return self.request_sync("GET", url, stream=stream, **kwargs)

    def post(self, url: str, stream: bool = False, is_async: bool = False, **kwargs):
        if is_async:
            return self.request_async("POST", url, stream=stream, **kwargs)
        return self.request_sync("POST", url, stream=stream, **kwargs)

    def put(self, url: str, stream: bool = False, is_async: bool = False, **kwargs):
        if is_async:
            return self.request_async("PUT", url, stream=stream, **kwargs)
        return self.request_sync("PUT", url, stream=stream, **kwargs)

    def patch(self, url: str, stream: bool = False, is_async: bool = False, **kwargs):
        if is_async:
            return self.request_async("PATCH", url, stream=stream, **kwargs)
        return self.request_sync("PATCH", url, stream=stream, **kwargs)

    def delete(self, url: str, stream: bool = False, is_async: bool = False, **kwargs):
        if is_async:
            return self.request_async("DELETE", url, stream=stream, **kwargs)
        return self.request_sync("DELETE", url, stream=stream, **kwargs)
