import httpx
from typing import Optional, Dict, Any
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
