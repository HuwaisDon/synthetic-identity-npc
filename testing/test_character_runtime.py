import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from character.loader import CharacterLoader
from core.cognition_pipeline import CognitionPipeline, TurnInput
from runtime.cogni_runtime import CogniRuntime
from api.routes import TurnRequest


def test_generic_character_runtime_loads_and_builds_intention(tmp_path):
    loader = CharacterLoader(source_dir=PROJECT_ROOT / "characters")
    character = loader.load("morgan")
    assert character.character_id == "morgan"
    assert character.goals[0].goal_id == "g_survival"

    runtime = CogniRuntime(state_store=None)
    loaded = runtime.load_character("morgan")
    assert loaded.character_id == "morgan"

    pipeline = CognitionPipeline(
        npc_id="morgan",
        npc_name="Morgan",
        character_brief=loaded.character_brief,
        character=loaded,
        state_store=None,
    )

    output = pipeline.process_turn(
        TurnInput(
            npc_id="morgan",
            player_message="Do you still believe in the old stories?",
            player_trust_level=0.6,
            threat_level=0.2,
            node_emotion_tags={"mem_001": ["fear"]},
            node_descriptions={"mem_001": "storm memory"},
        )
    )

    assert output.cognitive_summary is not None
    assert output.cognitive_state.predictive_state.chosen_strategy in {"disclose", "partial", "deflect", "lie", "silence"}
    assert output.cognitive_state.goal_state.dominant_goal is not None

    runtime.process_turn("morgan", TurnRequest(player_message="A small test", player_trust_level=0.6, threat_level=0.1))
    assert runtime.get_state("morgan")
