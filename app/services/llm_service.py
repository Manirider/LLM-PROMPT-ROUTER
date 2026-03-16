import logging
import httpx
import openai
from typing import Optional, Dict, Any
from config.settings import settings
from app.exceptions import LLMAPIError

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.use_ollama = settings.use_ollama
        self.openai_client = None
        if not self.use_ollama:
            self.openai_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

    async def get_chat_completion(
        self, 
        system_prompt: str, 
        user_message: str, 
        temperature: float = 0.7, 
        max_tokens: int = 512
    ) -> str:
        try:
            if self.use_ollama:
                return await self._get_ollama_completion(system_prompt, user_message, temperature, max_tokens)
            return await self._get_openai_completion(system_prompt, user_message, temperature, max_tokens)
        except Exception as exc:
            logger.exception("LLM completion request failed.")
            if isinstance(exc, LLMAPIError):
                raise
            raise LLMAPIError(f"LLM API failure: {exc}") from exc

    async def _get_ollama_completion(self, system_prompt: str, user_message: str, temperature: float, max_tokens: int) -> str:
        payload = {
            "model": settings.openai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "options": {"temperature": temperature, "max_tokens": max_tokens},
            "stream": False
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{settings.ollama_base_url}/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            if not data.get("choices"):
                return ""
            return str(data["choices"][0]["message"]["content"])

    async def _get_openai_completion(self, system_prompt: str, user_message: str, temperature: float, max_tokens: int) -> str:
        if self.openai_client is None:
            raise LLMAPIError("OpenAI client not initialized (USE_OLLAMA is True).")

        completion = await self.openai_client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        if not completion.choices:
            return ""
        return str(completion.choices[0].message.content)

llm_service = LLMService()
