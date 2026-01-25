# MineAI Python SDK

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://pypi.org/project/mineai/)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

The official Python SDK from [MineAI-Studio](https://studio.getmineai.site).

Powered by [http://getmineai.site/](http://getmineai.site/)

- [Github](https://github.com/OfficalMinecore/mineai-sdk)
- [Discord Server](https://discord.gg/fbfdwpHctb) – Join the server for Support.

## ✨ What's New in v1.0.0

- ✅ **Temperature Control** - Fine-tune response creativity and randomness
- ✅ **Max Tokens Limit** - Control response length
- ✅ **Retry on Failure** - Automatic retry with exponential backoff
- ✅ **Rate Limiting** - Built-in throttling detection and handling
- ✅ **Enhanced Error Handling** - Comprehensive error types and messages
- ✅ **Memory Support** - Database-backed conversation memory
- ✅ **Streaming Support** - Real-time response streaming
- ✅ **Async Support** - Full async/await compatibility

### Integrations:

- **Disocrd**: Go To [MineAI-Studio](https://studio.getmineai.site/) 

## Installation

```bash
pip install mineai
```

## Quick Start

### Sync Client

```python
from mineai import MineAI, Models

client = MineAI(api_key="YOUR_API_KEY")

response = client.chat.completions.create(
    model=Models.R3_RT_Y,
    messages=[
        {"role": "user", "content": "Hello MineAI!"}
    ]
)

print(response['choices'][0]['message']['content'])
```

### Async Client

```python
import asyncio
from mineai import AsyncMineAI, Models

async def main():
    client = AsyncMineAI(api_key="YOUR_API_KEY")
    
    response = await client.chat.completions.create(
        model=Models.R3_RT_Z,
        messages=[
            {"role": "user", "content": "Tell me a joke."}
        ]
    )
    print(response['choices'][0]['message']['content'])

asyncio.run(main())
```

### Streaming

```python
from mineai import MineAI, Models

client = MineAI(api_key="YOUR_API_KEY")

stream = client.chat.completions.create(
    model=Models.O1_FREE,
    messages=[
        {"role": "user", "content": "Write a long story."}
    ],
    stream=True
)

for chunk in stream:
    if 'choices' in chunk:
        content = chunk['choices'][0].get('delta', {}).get('content', '')
        print(content, end='', flush=True)
```

### Memory Support

```python
from mineai import MineAI, Models

client = MineAI(api_key="YOUR_API_KEY")

# Enable database-backed memory
response = client.chat.completions.create(
    model=Models.R3_RT_Y,
    messages=[
        {"role": "user", "content": "My name is John."}
    ],
    memory=True
)
```

### Advanced Parameters

#### Temperature Control

Control response randomness (0.0 = deterministic, 2.0 = very random):

```python
response = client.chat.completions.create(
    model=Models.R3_RT_Y,
    messages=[{"role": "user", "content": "Write a creative story."}],
    temperature=0.9  # Higher values = more creative/random
)
```

#### Max Tokens Limit

Limit the maximum tokens in the response:

```python
response = client.chat.completions.create(
    model=Models.O1_FREE,
    messages=[{"role": "user", "content": "Explain quantum physics."}],
    max_tokens=100  # Limit response to 100 tokens
)
```

#### Retry on Failure

Enable automatic retry with exponential backoff for failed requests:

```python
response = client.chat.completions.create(
    model=Models.R3_RT_Z,
    messages=[{"role": "user", "content": "Hello!"}],
    retry_on_failure=True  # Retries up to 3 times on 429/5xx errors
)
```

### Model-Specific Limits

#### mine:o1-free Memory Restriction

The free model (`mine:o1-free`) has a **15,000 tokens per month** limit for non-owner API keys:

- Regular users: 15K tokens/month limit enforced

```python
# Regular API key - subject to 15K/month limit
response = client.chat.completions.create(
    model=Models.O1_FREE,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Rate Limiting & Throttling

The API implements automatic throttling based on token usage patterns:

- **Rolling Window**: 10 seconds
- **Token Threshold**: 2,000 tokens
- **Model-Specific Delays**:
  - `mine:o1-free`: 3 second delay
  - `mine:r3-rt-y`: 2 second delay
  - `mine:r3-rt-z`: 1 second delay

```python
# The SDK automatically handles throttle delays
# Check response for throttle information
response = client.chat.completions.create(
    model=Models.R3_RT_Y,
    messages=[{"role": "user", "content": "Hello!"}]
)

# Throttle info available in response metadata
if response.get("throttle"):
    print(f"Request was throttled: {response.get('delay')}ms delay")
```

## Error Handling

The SDK provides custom exception classes for different API error scenarios:

```python
from mineai import MineAI, AuthenticationError, RateLimitError

try:
    client = MineAI(api_key="INVALID_KEY")
    client.chat.completions.create(...)
except AuthenticationError:
    print("Invalid API key provided.")
except RateLimitError:
    print("Rate limit exceeded.")
```

## Supported Models

- `Models.R3_RT_Y` (`mine:r3-rt-y`)
- `Models.R3_RT_Z` (`mine:r3-rt-z`)
- `Models.O1_FREE` (`mine:o1-free`)

## Testing

The comprehensive test script (`test_all_features.py`) is available on our [GitHub](https://github.com/OfficalMinecore/mineai-sdk-python).

Run the test suite:

```bash
export MINEAI_API_KEY="your-api-key"
python test_all_features.py
```

The test script validates all SDK features including completions, streaming, memory, temperature, max_tokens, retry logic, and rate limiting.

## Important Notes

### Free Tier Limitations (`mine:o1-free`)

- ❌ Memory feature not available
- ❌ Retry on failure not available  
- ✅ 15,000 tokens/month limit for regular API keys

### Paid Models (`mine:r3-rt-y`, `mine:r3-rt-z`)

- Require active credits and paid plan (Light, Medium, or Heavy)
- Full feature support including memory and retry

## Changelog

### v1.0.0 (2026-01-25)
- Added temperature, max_tokens, and retry_on_failure parameters
- Implemented automatic retry logic with exponential backoff
- Enhanced rate limiting and throttling support
- Improved error handling and messages
- Updated documentation with comprehensive examples
- Added comprehensive test suite

### v0.1.3
- Fixed memory implementation
- Improved error handling

## Issues & Support

- 🐛 Report bugs on [GitHub Issues](https://github.com/OfficalMinecore/mineai-sdk/issues)
- 💬 Get help on [Discord](https://discord.gg/fbfdwpHctb)
- 📧 Email support: support@getmineai.site

---

**Note**: Always use the latest version of the SDK for best performance and latest features.
