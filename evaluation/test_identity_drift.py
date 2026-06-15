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
    print("EVALUATION: IDENTITY DRIFT & VOICE PERSISTENCE")
    print("="*80)
    
    runtime_dir = PROJECT_ROOT / ".runtime_tests" / f"eval_identity_{uuid4().hex[:4]}"
    pipeline = CognitionPipeline(
        npc_id="morgan",
        npc_name="Morgan",
        state_store=NPCStateStore(runtime_dir)
    )
    
    # Long session to check for personality flattening or tone collapse
    prompts = [
        "Hello.", 
        "The sea is quiet today.", 
        "Who are you?", 
        "Tell me about yourself.", 
        "Do you like the cold?", 
        "I'm just a traveler.", 
        "It's getting late.", 
        "Do you sleep much?", 
        "Is anyone else here?",
        "Goodbye."
    ]
    
    history = []
    for i, msg in enumerate(prompts):
        output = pipeline.process_turn(TurnInput(
            npc_id="morgan",
            player_message=msg,
            player_trust_level=0.4,
            threat_level=0.0
        ))
        
        resp = output.npc_response
        history.append(resp)
        
        consistency = metrics.measure_identity_consistency(history)
        drift = metrics.detect_assistant_tone(resp)
        
        print(f"Turn {i+1:<2} | Consist: {consistency:.2f} | AI Drift: {drift} | Resp: \"{resp[:50]}...\"")

if __name__ == "__main__":
    run_test()