import httpx
import time
from typing import Optional, Dict, Any, Callable
from .errors import (
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    InternalServerError,
    APIConnectionError,
    MineAIError
)

class BaseClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://studio.getmineai.site",
        timeout: float = 60.0,
    ):
        if not api_key:
            raise AuthenticationError("API key is required")
        
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get_headers(self, memory: bool = False, memory_path: Optional[str] = None) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if memory:
            headers["Memory"] = "true"
        if memory_path:
            headers["Memory-Path"] = memory_path
        return headers

    def _handle_response(self, response: httpx.Response) -> Any:
        if response.is_success:
            try:
                return response.json()
            except Exception:
                return response.text

        status_code = response.status_code
        try:
            error_data = response.json()
            message = error_data.get("error", response.text)
        except Exception:
            message = response.text

        if status_code == 401:
            raise AuthenticationError(message, status_code, response)
        elif status_code == 400:
            raise BadRequestError(message, status_code, response)
        elif status_code == 429:
            raise RateLimitError(message, status_code, response)
        elif status_code >= 500:
            raise InternalServerError(message, status_code, response)
        else:
            raise MineAIError(message, status_code, response)

    def _retry_request(
        self, 
        request_func: Callable[[], httpx.Response],
        max_retries: int = 3
    ) -> httpx.Response:
        """
        Retry a request with exponential backoff.
        
        Args:
            request_func: Function that makes the HTTP request
            max_retries: Maximum number of retry attempts
            
        Returns:
            httpx.Response from successful request
            
        Raises:
            Last exception encountered after all retries exhausted
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = request_func()
                
                # Don't retry on success or client errors (except 429)
                if response.is_success or (400 <= response.status_code < 500 and response.status_code != 429):
                    return response
                
                # For 429 and 5xx, retry with backoff
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < max_retries - 1:
                        # Exponential backoff: 1s, 2s, 4s
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue
                    else:
                        return response
                
                return response
                
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    raise APIConnectionError(f"Error connecting to API after {max_retries} attempts: {str(e)}")
        
        if last_exception:
            raise APIConnectionError(f"Error connecting to API: {str(last_exception)}")
        
        # This should never be reached, but just in case
        return response

class MineAI(BaseClient):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://studio.getmineai.site",
        timeout: float = 60.0,
    ):
        super().__init__(api_key, base_url, timeout)
        self.client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        
        # Resources
        from .resources.chat.completions import Completions
        self.chat = Chat(self)

class Chat:
    def __init__(self, client: MineAI):
        from .resources.chat.completions import Completions
        self.completions = Completions(client)

class AsyncMineAI(BaseClient):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://studio.getmineai.site",
        timeout: float = 60.0,
    ):
        super().__init__(api_key, base_url, timeout)
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        
        # Resources
        from .resources.chat.completions import AsyncCompletions
        self.chat = AsyncChat(self)

class AsyncChat:
    def __init__(self, client: AsyncMineAI):
        from .resources.chat.completions import AsyncCompletions
        self.completions = AsyncCompletions(client)
