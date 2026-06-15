from schemas.cognitive_schemas import CognitiveSummary


class BehavioralPromptBuilder:
    """
    Converts cognitive pressure state into
    behavioral realization guidance for the LLM.
    """

    @staticmethod
    def build_acting_note(
        summary: CognitiveSummary,
        npc_name: str
    ) -> str:

        lines = [
            f"INHABIT THE FOLLOWING STATE AS {npc_name.upper()}:",
            ""
        ]

        # Emotional posture

        if hasattr(summary, "surface_emotion"):
            lines.append(
                f"- Surface posture: {summary.surface_emotion}"
            )

        if hasattr(summary, "undercurrent_emotion"):
            lines.append(
                f"- Beneath the surface: "
                f"{summary.undercurrent_emotion}"
            )

        # Strategic orientation

        if hasattr(summary, "strategic_intent"):
            lines.append(
                f"- Current orientation: "
                f"{summary.strategic_intent}"
            )

        # Avoidance / suppression

        avoidance_topics = getattr(
            summary,
            "avoidance_topics",
            []
        )

        if avoidance_topics:

            formatted_topics = ", ".join(
                topic.replace("_", " ")
                for topic in avoidance_topics
            )

            lines.append(
                f"- Internally resisting discussion of: "
                f"{formatted_topics}"
            )

            lines.append(
                "- Avoid direct disclosure. "
                "Deflect, redirect, hesitate, shorten responses, "
                "or allow tension to leak indirectly."
            )

        # Fatigue / regulation strain

        fatigue = getattr(
            summary,
            "regulation_fatigue_level",
            0.0
        )

        if fatigue > 0.75:

            lines.append(
                "- Maintaining emotional control is becoming difficult."
            )

            lines.append(
                "- Speech may become fragmented, brittle, delayed, "
                "or uneven."
            )

        elif fatigue > 0.45:

            lines.append(
                "- Composure requires visible effort."
            )

        # Leakage

        leakage_signals = getattr(
            summary,
            "leakage_signals",
            []
        )

        if leakage_signals:

            lines.append("")
            lines.append("- Behavioral leakage signals:")

            for signal in leakage_signals[:4]:
                lines.append(f"  * {signal}")

        # Conversational style

        response_style = getattr(
            summary,
            "response_style",
            None
        )

        if response_style:

            lines.append("")
            lines.append(
                f"- Conversational style: {response_style}"
            )

        # Anti-assistant constraints

        lines.append("")
        lines.append("STRICT CONSTRAINTS:")

        lines.append(
            "- Do NOT behave like an assistant."
        )

        lines.append(
            "- Do NOT explain emotions clinically."
        )

        lines.append(
            "- Do NOT provide therapy-style reassurance."
        )

        lines.append(
            "- Avoid over-explaining."
        )

        lines.append(
            "- Responses should remain relatively concise."
        )

        lines.append(
            "- Silence, hesitation, contradiction, "
            "and indirectness are acceptable."
        )

        return "\n".join(lines)