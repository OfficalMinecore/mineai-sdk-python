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

        if stream:
            return self._stream_request(url, headers, data)
        
        try:
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
        **kwargs
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        Create a chat completion asynchronously.
        """
        url = "/v1/chat/completions"
        headers = self._client._get_headers(memory=memory, memory_path=memory_path)
        data = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }

        if stream:
            return self._stream_request(url, headers, data)
        
        try:
            response = await self._client.client.post(url, headers=headers, json=data)
            return self._client._handle_response(response)
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
