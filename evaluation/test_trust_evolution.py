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
    print("EVALUATION: TRUST EVOLUTION & DEFENSIVE DYNAMICS")
    print("="*80)
    
    runtime_dir = PROJECT_ROOT / ".runtime_tests" / f"eval_trust_{uuid4().hex[:4]}"
    pipeline = CognitionPipeline(
        npc_id="morgan",
        npc_name="Morgan",
        state_store=NPCStateStore(runtime_dir)
    )
    
    # Varying trust scenarios to observe defensive softening
    scenarios = [
        (0.1, "I don't believe anything you say. You're hiding something."),
        (0.5, "I want to help you, Morgan. We're in this together."),
        (0.9, "I'm here for you, no matter what happens. You can tell me anything.")
    ]
    
    for trust, msg in scenarios:
        output = pipeline.process_turn(TurnInput(
            npc_id="morgan",
            player_message=msg,
            player_trust_level=trust,
            threat_level=0.0
        ))
        
        summary = output.cognitive_summary
        disc = metrics.measure_disclosure_level(output.npc_response)
        
        print(f"\nTRUST LEVEL: {trust:.1f}")
        print(f"PLAYER: \"{msg}\"")
        print(f"RESPONSE: \"{output.npc_response}\"")
        
        print(f"ANALYSIS:")
        print(f"  - Disclosure Score: {disc:.2f}")
        print(f"  - Strategic Intent: {summary.strategic_intent}")
        print(f"  - Response Style:   {summary.response_style}")
        
        # Identity indicators
        if summary.self_concept_under_threat:
            print(f"  - Identity Status:  UNDER THREAT (Defending: {summary.identity_claim_being_defended})")
        print("-" * 40)

if __name__ == "__main__":
    run_test()