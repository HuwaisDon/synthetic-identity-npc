import os
import time

from dotenv import load_dotenv
from google import genai
from loguru import logger

load_dotenv()


class GeminiClient:

    def __init__(self):

        self.api_key = os.getenv("GEMINI_API_KEY")

        self.model = os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash"
        )

        if not self.api_key:
            self.client = None
            logger.warning(
                "[Gemini] GEMINI_API_KEY not set; using fallback responses"
            )
        else:
            self.client = genai.Client(
                api_key=self.api_key
            )

    def generate_response(
        self,
        prompt: str | None = None,
        acting_note: str | None = None,
        system_instruction: str | None = None,
        history: list | None = None,
    ) -> str:

        start_time = time.time()

        prompt = prompt or acting_note or ""

        if self.client is None:
            logger.warning("[Gemini] No client configured; returning fallback")
            return (
                "... Morgan looks away for a long moment, "
                "jaw tight, unable to answer directly ..."
            )

        conversation = ""

        # prior turns

        if history:

            for item in history:

                role = item.get("role", "user")
                content = item.get("content", "")

                conversation += f"{role.upper()}: {content}\n"

        full_prompt = f"""
{system_instruction or ""}

{conversation}

CURRENT INTERNAL STATE:
{prompt}

Respond ONLY as the character.
Remain psychologically consistent.
Do not narrate system state.
Do not behave like an AI assistant.
"""

        try:

            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
            )

            latency = time.time() - start_time

            logger.info(
                f"[Gemini] "
                f"model={self.model} "
                f"latency={latency:.2f}s "
                f"prompt_chars={len(full_prompt)} "
                f"history_items={len(history) if history else 0}"
            )

            return response.text.strip()

        except Exception as e:

            logger.error(f"[Gemini ERROR] {e}")

            return (
                "... Morgan looks away for a long moment, "
                "jaw tight, unable to answer directly ..."
            )
