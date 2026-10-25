import json
import logging
import time
from dataclasses import dataclass

from app.config import settings

log = logging.getLogger(__name__)


class LLMError(Exception):
    """ყველა პროვაიდერი ჩავარდა."""


@dataclass
class LLMResult:
    text: str
    provider: str
    latency_s: float


class GeminiProvider:
    name = "gemini"

    def __init__(self) -> None:
        from google import genai
        from google.genai import types
        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(
                timeout=int(settings.llm_timeout_s * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
    def generate(self, prompt: str, system: str | None, json_mode: bool) -> str:
        from google.genai import types
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            response_mime_type="application/json" if json_mode else None,
        )
        r = self.client.models.generate_content(
            model=settings.gemini_model, contents=prompt, config=cfg
        )
        return r.text or ""


class GroqProvider:
    name = "groq"

    def __init__(self) -> None:
        from groq import Groq
        self.client = Groq(api_key=settings.groq_api_key,
                           timeout=settings.llm_timeout_s)

    def generate(self, prompt: str, system: str | None, json_mode: bool) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        r = self.client.chat.completions.create(
            model=settings.groq_model, messages=messages, temperature=0.0, **kwargs
        )
        return r.choices[0].message.content or ""


class LLMClient:
    def __init__(self, providers: list | None = None, order: tuple[str, ...] = ("groq", "gemini")) -> None:
        self.providers = providers if providers is not None else self._default(order)
        if not self.providers:
            raise LLMError("არცერთი LLM გასაღები არ არის მითითებული (.env)")
        
    @staticmethod
    def _default(order: tuple[str, ...] = ("groq", "gemini")) -> list:
        makers = {
            "gemini": (settings.gemini_api_key, GeminiProvider),
            "groq": (settings.groq_api_key, GroqProvider),
        }
        return [makers[n][1]() for n in order if makers[n][0]]

    def generate(self, prompt: str, system: str | None = None,
                 json_mode: bool = False) -> LLMResult:
        errors = []
        for provider in self.providers:
            for attempt in range(settings.llm_max_retries):
                t0 = time.time()
                try:
                    text = provider.generate(prompt, system, json_mode)
                    return LLMResult(text, provider.name, time.time() - t0)
                except Exception as e:                       # noqa: BLE001
                    errors.append(f"{provider.name}#{attempt + 1}: {e}")
                    log.warning("LLM შეცდომა %s: %s", provider.name, e)
                    if not self._retryable(e):
                        break                               # არასწორი მოთხოვნა, ხელახლა არ ვცდით
                    time.sleep(2 ** attempt)                # 1, 2, 4 წმ
        raise LLMError("ყველა პროვაიდერი ჩავარდა: " + " | ".join(errors[-4:]))

    def generate_json(self, prompt: str, system: str | None = None) -> dict:
        res = self.generate(prompt, system, json_mode=True)
        text = res.text.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"LLM-მა არასწორი JSON დააბრუნა: {text[:200]}") from e

    @staticmethod
    def _retryable(e: Exception) -> bool:
        msg = str(e).lower()
        return any(s in msg for s in ("429", "503", "500", "timeout", "unavailable",
                                      "rate", "overloaded", "deadline"))