import os
import time
import requests

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class OpenRouterClient:

    def __init__(self):

        self.api_key = os.getenv("OPENROUTER_API_KEY")

        self.model = os.getenv(
            "OPENROUTER_MODEL",
            "qwen/qwen3-32b:free"
        )

        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def generate_response(
        self,
        prompt: str | None = None,
        acting_note: str | None = None,
        system_instruction: str | None = None,
        history: list | None = None,
    ) -> str:

        start_time = time.time()

        prompt = prompt or acting_note or ""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "SyntheticIdentityNPC",
        }

        messages = [
            {
                "role": "system",
                "content": (
                    system_instruction
                    or
                    (
                        "You are realizing the behavior of a persistent synthetic individual. "
                        "Do not behave like an assistant. "
                        "Remain psychologically grounded, emotionally constrained, "
                        "and behaviorally coherent."
                    )
                )
            }
        ]

        # Prior conversation history

        if history:

            for item in history:

                if isinstance(item, dict):

                    role = item.get("role")
                    content = item.get("content")

                    if role and content:

                        messages.append({
                            "role": role,
                            "content": content,
                        })

        # Current player message

        messages.append({
            "role": "user",
            "content": prompt
        })

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.85,
            "max_tokens": 220,
        }

        try:

            response = requests.post(
                self.url,
                headers=headers,
                json=payload,
                timeout=60,
            )

            latency = time.time() - start_time

            logger.info(
                f"[OpenRouter] "
                f"model={self.model} "
                f"status={response.status_code} "
                f"latency={latency:.2f}s "
                f"prompt_chars={len(prompt)} "
                f"history_items={len(history) if history else 0}"
            )

            response.raise_for_status()

            data = response.json()

            return data["choices"][0]["message"]["content"]

        except Exception as e:
            if 'response' in locals():
                logger.error(response.text)
            logger.error(f"[OpenRouter ERROR] {e}")

            return (
                "... Morgan looks away for a long moment, "
                "jaw tight, unable to answer directly ..."
            )