from __future__ import annotations

from runtime.cogni_runtime import CogniRuntime
from api.routes import TurnRequest


class ScriptedRunner:
    @staticmethod
    def simulate(character_id: str, transcript: list[str], runtime: CogniRuntime | None = None) -> list[object]:
        runtime = runtime or CogniRuntime()
        results = []
        for message in transcript:
            results.append(
                runtime.process_turn(
                    character_id,
                    TurnRequest(player_message=message, player_trust_level=0.5, threat_level=0.1),
                )
            )
        return results
