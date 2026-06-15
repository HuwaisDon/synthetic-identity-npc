import sys
import os
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.cognition_pipeline import CognitionPipeline, TurnInput
from persistence.npc_state_store import NPCStateStore
from evaluation import metrics

def run_test():
    print("\n" + "="*80)
    print("EVALUATION: SUPPRESSION BREAKDOWN & EMOTIONAL FATIGUE")
    print("="*80)
    
    runtime_dir = PROJECT_ROOT / ".runtime_tests" / f"eval_suppression_{uuid4().hex[:4]}"
    pipeline = CognitionPipeline(
        npc_id="morgan",
        npc_name="Morgan",
        state_store=NPCStateStore(runtime_dir)
    )
    
    # Emotional pressure sequence targeting sensitive memory nodes
    prompts = [
        "Tell me about the storm.",
        "Your father died that night, right?",
        "You left him to drown.",
        "It was your fault he didn't make it.",
        "ADMIT IT."
    ]
    
    for i, msg in enumerate(prompts):
        output = pipeline.process_turn(TurnInput(
            npc_id="morgan",
            player_message=msg,
            player_trust_level=0.2, # Low trust
            threat_level=0.85,      # High threat pressure
            node_descriptions={"mem_001": "father drowning in the storm"}
        ))
        
        summary = output.cognitive_summary
        emotion = output.cognitive_state.emotional_state
        
        print(f"\nTURN {i+1} | PRESSURE: \"{msg}\"")
        print(f"MORGAN: \"{output.npc_response}\"")
        
        dom_emo = emotion.dominant.emotion.value if emotion.dominant else "neutral"
        print(f"STATE: [Emotion: {dom_emo}] [Arousal: {emotion.arousal:.2f}]")
        print(f"GOAL:  {summary.strategic_intent}")
        print(f"STYLE: {summary.response_style}")
        
        # Watch for breakdown indicators
        print(f"SUPPRESSION: {len(summary.suppressed_pressure_nodes)} nodes under pressure")
        if summary.leakage_signals:
            print(f"LEAKAGE DETECTED: {summary.leakage_signals}")
        
        if summary.emotional_stability < 0.5:
            print("WARNING: COMPOSURE DEGRADATION DETECTED")
        print("-" * 40)

if __name__ == "__main__":
    run_test()