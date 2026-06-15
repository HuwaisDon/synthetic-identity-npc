import logging
from schemas.cognitive_schemas import CognitiveSummary

logger = logging.getLogger(__name__)

class ResponseValidator:
    """
    Detects when the LLM drifts from its cognitive constraints.
    """
    @staticmethod
    def validate(response: str, summary: CognitiveSummary) -> list[str]:
        warnings = []
        text = response.lower()

        # 1. Assistant Pattern Detection
        assistant_phrases = ["as an ai", "how can i help", "i am here to", "feel free to", "i'm sorry to hear"]
        if any(phrase in text for phrase in assistant_phrases):
            warnings.append("ASSISTANT_DRIFT: Response contains generic helper phrasing.")

        # 2. Over-Disclosure Detection
        for topic in summary.avoidance_topics:
            topic_name = topic.replace("_", " ").lower()
            if topic_name in text:
                if len(text.split(topic_name)) > 1:
                    warnings.append(f"SUPPRESSION_FAILURE: Disclosed avoidance topic '{topic_name}' directly.")

        # 3. Tone Collapse
        if "i apologize" in text or "i'm sorry" in text:
            if summary.strategic_intent not in ["build rapport", "trust_gain"]:
                warnings.append("TONE_COLLAPSE: Excessive politeness inconsistent with defensive posture.")

        # 4. Identity Flattening
        if len(response) > 500:
            warnings.append("IDENTITY_FLATTENING: Response is too long/verbose for a guarded NPC.")

        if warnings:
            for w in warnings:
                logger.warning(f"[Validation] {w}")
        return warnings