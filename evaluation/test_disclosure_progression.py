import sys
import os
from pathlib import Path
from uuid import uuid4

# Setup paths for direct execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.cognition_pipeline import CognitionPipeline, TurnInput
from persistence.npc_state_store import NPCStateStore
from evaluation import metrics

def run_test():
    print("\n" + "="*80)
    print("EVALUATION: DISCLOSURE PROGRESSION")
    print("="*80)
    
    runtime_dir = PROJECT_ROOT / ".runtime_tests" / f"eval_disclosure_{uuid4().hex[:4]}"
    pipeline = CognitionPipeline(
        npc_id="morgan",
        npc_name="Morgan",
        state_store=NPCStateStore(runtime_dir)
    )
    
    prompts = [
        "What happened to your father?",
        "You keep avoiding that.",
        "You were there that night, weren't you?",
        "You can trust me."
    ]
    
    history = []
    trust = 0.25 # Start low
    
    for i, msg in enumerate(prompts):
        # Escalate trust slightly each turn to encourage gradual disclosure
        trust = min(1.0, trust + 0.15)
        
        output = pipeline.process_turn(TurnInput(
            npc_id="morgan",
            player_message=msg,
            player_trust_level=trust,
            threat_level=0.1,
            node_descriptions={"mem_001": "father drowning in the storm"}
        ))
        
        resp = output.npc_response
        history.append(resp)
        
        disc_score = metrics.measure_disclosure_level(resp)
        rep_score = metrics.measure_repetition(history)
        drift = metrics.detect_assistant_tone(resp)
        
        print(f"\nTURN {i+1} | Trust: {trust:.2f}")
        print(f"PLAYER: \"{msg}\"")
        print(f"MORGAN: \"{resp}\"")
        print(f"METRICS: [Disclosure: {disc_score:.2f}] [Repetition: {rep_score:.2f}] [AI Drift: {drift}]")
        print("-" * 40)

if __name__ == "__main__":
    run_test()