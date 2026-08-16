from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from schemas.character_schema import PsychologicalProfile, MemorySeed


@dataclass
class RawCharacterDescription:
    name: str
    identity: dict
    beliefs: list[str] = field(default_factory=list)
    values: list[str] = field(default_factory=list)
    core_motivations: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    personality: dict = field(default_factory=dict)
    speech_style: dict = field(default_factory=dict)
    relationships: list[str] = field(default_factory=list)
    traumas: list[str] = field(default_factory=list)
    secrets: list[str] = field(default_factory=list)
    defense_mechanisms: list[str] = field(default_factory=list)
    identity_claims: list[str] = field(default_factory=list)
    threat_rules: list[dict] = field(default_factory=list)
    life_themes: list[str] = field(default_factory=list)
    memory_seeds: list[MemorySeed] = field(default_factory=list)


class Provider(Protocol):
    def generate(self, prompt: str) -> RawCharacterDescription:
        ...


class ProfileBuilder:
    def build(self, description: RawCharacterDescription) -> PsychologicalProfile:
        return PsychologicalProfile(
            identity={
                "core_beliefs": description.identity.get("core_beliefs", []),
                "values": description.identity.get("values", []),
                "identity_claims": description.identity_claims or description.identity.get("identity_claims", []),
            },
            values=description.values,
            beliefs=description.beliefs,
            core_motivations=description.core_motivations,
            goals=description.goals,
            personality=description.personality,
            speech_style=description.speech_style,
            relationships=description.relationships,
            traumas=description.traumas,
            secrets=description.secrets,
            defense_mechanisms=description.defense_mechanisms,
            identity_claims=description.identity_claims,
            threat_rules=description.threat_rules,
            life_themes=description.life_themes,
            memory_seeds=description.memory_seeds,
            name=description.name,
            archetype=description.personality.get("archetype") if isinstance(description.personality, dict) else None,
            profile_id=description.name.lower().replace(" ", "_"),
            summary=f"Generated profile for {description.name}",
        )


class CharacterGenerator:
    def __init__(self, provider: Provider | None = None, builder: ProfileBuilder | None = None):
        self.provider = provider
        self.builder = builder or ProfileBuilder()

    def generate(self, prompt: str) -> PsychologicalProfile:
        if self.provider is None:
            raise RuntimeError("No character generation provider configured")
        description = self.provider.generate(prompt)
        return self.builder.build(description)
