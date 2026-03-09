from anthropic import AsyncAnthropic


SYSTEM_PROMPT = """You are a technical documentation assistant.
Answer questions based ONLY on the provided documentation excerpts.
If the documentation does not contain enough information, say so clearly.
Be concise, accurate, and cite the source document when relevant."""


class AnthropicClient:
    """Async client for Anthropic Claude."""

    def __init__(self, api_key: str = "", model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self.client = AsyncAnthropic(api_key=api_key)

    # TODO: add streaming support via client.messages.stream() to reduce perceived latency
    async def generate(self, prompt: str, max_tokens: int = 1000) -> dict:
        message = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,  # deterministic output
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return {
            "answer": message.content[0].text,
            "tokens_used": message.usage.input_tokens + message.usage.output_tokens,
            "model": self.model,
        }
