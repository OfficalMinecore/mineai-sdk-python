class MineAIError(Exception):
    """Base class for all MineAI SDK errors."""
    def __init__(self, message: str, status_code: int = None, response: any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response

class AuthenticationError(MineAIError):
    """Raised when the API key is invalid or missing."""
    pass

class BadRequestError(MineAIError):
    """Raised when the request is invalid."""
    pass

class RateLimitError(MineAIError):
    """Raised when the rate limit is exceeded."""
    pass

class InternalServerError(MineAIError):
    """Raised when the server encounters an error."""
    pass

class APIConnectionError(MineAIError):
    """Raised when there is a problem connecting to the API."""
    pass
