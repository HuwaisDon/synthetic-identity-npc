from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from schemas.cognitive_schemas import GoalType


@dataclass
class Intention:
    dominant_drive: Optional[GoalType] = None
    disposition: str = "neutral"
    intensity: float = 0.0
    target_topics: list[str] = field(default_factory=list)
    supported_modalities: list[str] = field(default_factory=lambda: ["dialogue"])


@dataclass
class ExpressionRequest:
    intention: Intention
    modality: str = "dialogue"
