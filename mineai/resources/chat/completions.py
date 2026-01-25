import json
from typing import List, Dict, Any, Optional, Union, Iterator, AsyncIterator
import httpx
from ...errors import APIConnectionError

class Completions:
    def __init__(self, client):
        self._client = client

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        memory: bool = False,
        memory_path: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retry_on_failure: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Create a chat completion.
        
        Args:
            model: The ID of the model to use.
            messages: A list of messages comprising the conversation.
            stream: If True, partial message deltas will be sent as SSE.
            memory: If True, enables conversation memory.
            memory_path: Optional path for file-based memory.
            temperature: Controls randomness (0.0 to 2.0). Higher = more random.
            max_tokens: Maximum tokens to generate in the response.
            retry_on_failure: If True, retry failed requests with exponential backoff.
            **kwargs: Additional parameters to pass to the API.
        """
        url = "/v1/chat/completions"
        headers = self._client._get_headers(memory=memory, memory_path=memory_path)
        data = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        # Add optional parameters if provided
        if temperature is not None:
            data["temperature"] = temperature
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if retry_on_failure:
            data["retry_on_failure"] = retry_on_failure

        if stream:
            return self._stream_request(url, headers, data)
        
        try:
            if retry_on_failure:
                response = self._client._retry_request(
                    lambda: self._client.client.post(url, headers=headers, json=data)
                )
            else:
                response = self._client.client.post(url, headers=headers, json=data)
            return self._client._handle_response(response)
        except httpx.RequestError as e:
            raise APIConnectionError(f"Error connecting to API: {str(e)}")

    def _stream_request(self, url: str, headers: Dict[str, str], data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        try:
            with self._client.client.stream("POST", url, headers=headers, json=data) as response:
                if not response.is_success:
                    response.read()
                    self._client._handle_response(response)
                
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        content = line[6:].strip()
                        if content == "[DONE]":
                            break
                        try:
                            yield json.loads(content)
                        except json.JSONDecodeError:
                            continue
        except httpx.RequestError as e:
            raise APIConnectionError(f"Error connecting to API: {str(e)}")

class AsyncCompletions:
    def __init__(self, client):
        self._client = client

    async def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        memory: bool = False,
        memory_path: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retry_on_failure: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        Create a chat completion asynchronously.
        
        Args:
            model: The ID of the model to use.
            messages: A list of messages comprising the conversation.
            stream: If True, partial message deltas will be sent as SSE.
            memory: If True, enables conversation memory.
            memory_path: Optional path for file-based memory.
            temperature: Controls randomness (0.0 to 2.0). Higher = more random.
            max_tokens: Maximum tokens to generate in the response.
            retry_on_failure: If True, retry failed requests with exponential backoff.
            **kwargs: Additional parameters to pass to the API.
        """
        url = "/v1/chat/completions"
        headers = self._client._get_headers(memory=memory, memory_path=memory_path)
        data = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        # Add optional parameters if provided
        if temperature is not None:
            data["temperature"] = temperature
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if retry_on_failure:
            data["retry_on_failure"] = retry_on_failure

        if stream:
            return self._stream_request(url, headers, data)
        
        try:
            # For async, implement simple retry logic inline
            max_retries = 3 if retry_on_failure else 1
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    response = await self._client.client.post(url, headers=headers, json=data)
                    
                    # Don't retry on success or client errors (except 429)
                    if response.is_success or (400 <= response.status_code < 500 and response.status_code != 429):
                        return self._client._handle_response(response)
                    
                    # For 429 and 5xx, retry with backoff
                    if (response.status_code == 429 or response.status_code >= 500) and attempt < max_retries - 1:
                        import asyncio
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                        continue
                    
                    return self._client._handle_response(response)
                    
                except httpx.RequestError as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        import asyncio
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                    else:
                        raise APIConnectionError(f"Error connecting to API after {max_retries} attempts: {str(e)}")
            
            if last_error:
                raise APIConnectionError(f"Error connecting to API: {str(last_error)}")
                
        except httpx.RequestError as e:
            raise APIConnectionError(f"Error connecting to API: {str(e)}")

    async def _stream_request(self, url: str, headers: Dict[str, str], data: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        try:
            async with self._client.client.stream("POST", url, headers=headers, json=data) as response:
                if not response.is_success:
                    await response.aread()
                    self._client._handle_response(response)
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        content = line[6:].strip()
                        if content == "[DONE]":
                            break
                        try:
                            yield json.loads(content)
                        except json.JSONDecodeError:
                            continue
        except httpx.RequestError as e:
            raise APIConnectionError(f"Error connecting to API: {str(e)}")
