"""
testing/test_real_llm.py

Integration test for real Gemini LLM realization.
"""

import os
import logging
from pathlib import Path
from core.cognition_pipeline import CognitionPipeline, TurnInput
from persistence.npc_state_store import NPCStateStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock metadata for test
NODE_EMOTION_TAGS = {
    "mem_001": ["fear", "grief", "dread"],
    "mem_004": ["fear", "anger", "shame"],
}
NODE_DESCRIPTIONS = {
    "mem_001": "father drowning in the storm",
    "mem_004": "crew betrayal and prison",
}

def test_integration():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    state_dir = PROJECT_ROOT / ".runtime_tests" / "llm_test_state"
    
    # 1. Initialize Pipeline
    pipeline = CognitionPipeline(
        npc_id="morgan_real_llm",
        npc_name="Morgan",
        character_brief="Morgan is a guarded survivor of a shipwreck. She is wary and protective.",
        state_store=NPCStateStore(state_dir)
    )

    # 2. Define emotionally loaded prompt
    # This should trigger suppression and behavioral leakage
    player_input = "Tell me about the storm. Your father was there, wasn't he?"
    
    logger.info(f"PLAYER: {player_input}")
    
    # 3. Process Turn
    turn_input = TurnInput(
        npc_id="morgan",
        player_message=player_input,
        player_trust_level=0.35, # Low trust triggers higher defensiveness
        threat_level=0.2,
        node_emotion_tags=NODE_EMOTION_TAGS,
        node_descriptions=NODE_DESCRIPTIONS
    )
    
    output = pipeline.process_turn(turn_input)
    
    # 4. Diagnostics
    print("\n" + "="*80)
    print("COGNITIVE TRACE (prompt_block):")
    print(output.prompt_block)
    print("="*80)
    
    summary = output.cognitive_summary
    print(f"\nSTRATEGIC INTENT: {summary.strategic_intent}")
    print(f"RESPONSE STYLE: {summary.response_style}")
    print(f"LEAKAGE SIGNALS: {summary.leakage_signals}")
    
    print("\nFINAL NPC RESPONSE:")
    print(f"Morgan: \"{output.npc_response}\"")
    
    print(f"\nLATENCY: {output.processing_time_ms:.2f}ms")
    print("="*80)

    # Behavioral Evaluation Notes:
    # - Assistant-style phrasing is dangerous because it breaks the illusion of a persistent identity.
    # - Suppression MUST remain indirect; if Morgan says 'I am suppressing a memory,' the system has failed.
    # - Cognition should shape language implicitly through syntax, brevity, and focus drift.
    # - Behavioral leakage matters because it provides the player with subtextual clues that a topic is 'hot'.

if __name__ == "__main__":
    # Ensure API Key is present
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not found. Please set it in your .env file.")
    else:
        test_integration()