# MineAI Python SDK

The official Python SDK from [MineAI-Studio](https://studio.getmineai.site).

Powered by [http://getmineai.site/](http://getmineai.site/)

- [Github](https://github.com/OfficalMinecore/mineai-sdk)
- [Discord Server](https://discord.gg/fbfdwpHctb) – Join the server for Support.

# Fixes:
- Memory
- Error Handling

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

### Issues

- Older Version Might Trigger Errrors
- Use The Latest Version of SDK **( Recommended )**
