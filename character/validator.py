from __future__ import annotations

from typing import Optional

from schemas.character_schema import PsychologicalProfile


class CharacterValidator:
    """Validate a PsychologicalProfile before compilation."""

    def validate(self, profile: PsychologicalProfile) -> list[str]:
        errors: list[str] = []

        if not profile.identity_claims:
            errors.append("identity_claims: at least one identity claim is required")

        if not profile.goals:
            errors.append("goals: at least one goal is required")

        if not profile.memory_seeds:
            errors.append("memories: at least one memory seed is required")

        if profile.identity.get("core_beliefs") in (None, []) and not profile.beliefs:
            errors.append("identity.core_beliefs: at least one core belief is required")

        seen_ids: set[str] = set()
        for seed in profile.memory_seeds:
            if not seed.seed_id:
                errors.append("memory_seeds: each memory seed requires a non-empty seed_id")
                continue
            if seed.seed_id in seen_ids:
                errors.append(f"memory_seeds: duplicate seed_id '{seed.seed_id}'")
            seen_ids.add(seed.seed_id)

        for idx, rule in enumerate(profile.threat_rules or []):
            if not rule.get("trigger_concepts"):
                errors.append(f"threat_rules[{idx}].trigger_concepts: at least one concept is required")
            if not isinstance(rule.get("threatened_claim_idx"), int):
                errors.append(f"threat_rules[{idx}].threatened_claim_idx: must be an integer")

        if profile.identity_claims and profile.threat_rules:
            for index, rule in enumerate(profile.threat_rules):
                claim_idx = rule.get("threatened_claim_idx")
                if isinstance(claim_idx, int) and claim_idx >= len(profile.identity_claims):
                    errors.append(
                        f"threat_rules[{index}].threatened_claim_idx: {claim_idx} is outside identity_claims range"
                    )

        return errors
