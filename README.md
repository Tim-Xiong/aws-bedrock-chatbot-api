# Conversational AI Assistant (Amazon Bedrock + Llama 4 Maverick)

This project provides a conversational AI assistant using Amazon Bedrock with the Llama 4 Maverick 17B Instruct model. It supports context management, error handling, and both CLI and programmatic usage.

---

## Installation

- Requires Python 3.8+
- Install dependencies:
  ```bash
  pip install boto3 botocore
  ```
- AWS credentials must be configured (see [AWS docs](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html)).

---

## CLI Usage

Run the assistant interactively:

```bash
python chat.py [--region REGION] [--model MODEL_ID] [--max-tokens N] [--temperature T] [--conversation-id ID]
```

**Options:**
- `--region`         AWS region (default: us-east-1)
- `--model`          Model or inference profile ID (default: us.meta.llama4-maverick-17b-instruct-v1:0)
- `--max-tokens`     Max tokens for generation (default: 512)
- `--temperature`    Sampling temperature (default: 0.7)
- `--conversation-id` Continue an existing conversation by ID

**Commands in CLI:**
- `quit`   Exit
- `info`   Show conversation details
- `reset`  Clear conversation history

---

## Programmatic API

### Main Classes

- `ConversationalAI(config=None, memory_store=None)`
  - Main entry point. Handles chat, context, and Bedrock API.
- `BedrockConfig` — Model/configuration options
- `InMemoryStore` — In-memory conversation storage
- `ConversationContext` — Conversation state

### Example: Basic Chat

```python
from chat import ConversationalAI

ai = ConversationalAI()
result = ai.chat("Hello!")
print(result["response"])
```

### Example: Continue a Conversation

```python
conv_id = None
ai = ConversationalAI()

# First message
resp1 = ai.chat("Hi!", conversation_id=conv_id)
conv_id = resp1["conversation_id"]

# Next message
resp2 = ai.chat("Tell me a joke.", conversation_id=conv_id)
print(resp2["response"])
```

### Example: Get/Reset Conversation Info

```python
info = ai.get_conversation_info(conv_id)
ai.reset_conversation(conv_id)
```

---

## Public API Methods

- `ConversationalAI.chat(message, conversation_id=None)` → `{response, conversation_id, message_count, timestamp}`
- `ConversationalAI.get_conversation_info(conversation_id)` → dict or None
- `ConversationalAI.reset_conversation(conversation_id)` → bool

---

## Testing

Tests are in `test.py` and use `pytest` and `unittest.mock` for AWS mocking.

---

## Notes
- Requires AWS Bedrock access and correct permissions.
- Handles context, retries, and error cases robustly.
- All context is in-memory by default (see `InMemoryStore`).
