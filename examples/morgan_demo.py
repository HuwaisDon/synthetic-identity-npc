"""
examples/morgan_demo.py

Working Example: Full Cognitive Pipeline with Morgan

This demonstrates:
1. Pipeline initialization
2. Processing turns with varying emotional/threat content
3. Observing how cognitive state evolves across turns
4. How different conversation directions produce different behavioral pressure

Run: python -m examples.morgan_demo

Note: Uses stub LLM caller. Replace with real LLM for production.
"""

from __future__ import annotations
import sys
import os
import logging
from pathlib import Path

# Make root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)  # Set to DEBUG for full trace

from core.cognition_pipeline import CognitionPipeline, TurnInput
from persistence.npc_state_store import NPCStateStore


# ─────────────────────────────────────────────
# MORGAN'S NODE EMOTION TAGS
# Maps memory node IDs to the emotions they carry.
# In production: stored in your memory graph.
# ─────────────────────────────────────────────

MORGAN_NODE_EMOTION_TAGS = {
    "childhood_with_kara":       ["longing", "grief"],
    "last_good_night":           ["grief", "longing"],
    "karas_body":                ["fear", "shame", "guilt"],
    "river_location":            ["fear", "dread"],
    "the_night_it_happened":     ["fear", "shame", "guilt"],
    "helplessness_memory":       ["shame", "grief"],
    "begging_scene":             ["shame", "fear"],
    "crying_alone":              ["grief", "shame"],
    "the_knife":                 ["fear", "guilt"],
    "self_as_survivor":          ["relief"],
    "self_as_protector":         ["longing"],
    "connection_memory":         ["longing"],
    "witness_old_maret":         ["fear", "shame"],
    "grief_for_kara":            ["grief"],
}

MORGAN_NODE_DESCRIPTIONS = {
    "childhood_with_kara":       "summers spent together before everything changed",
    "last_good_night":           "the last evening they were happy, before the end",
    "karas_body":                "what Morgan found at the river",
    "river_location":            "the place she has not returned to",
    "the_night_it_happened":     "the sequence of events Morgan does not speak about",
    "helplessness_memory":       "the moment she knew she couldn't stop what was coming",
    "begging_scene":             "the one time she asked for help and no one came",
    "crying_alone":              "the grief she has never shown anyone",
    "the_knife":                 "the object she disposed of",
    "self_as_survivor":          "the way she thinks of herself: someone who endured",
    "self_as_protector":         "the version of herself she tries to believe",
    "connection_memory":         "moments of real closeness she cannot access now",
    "witness_old_maret":         "the old woman who may have seen something",
    "grief_for_kara":            "the grief she suppresses every day",
}


def run_demo():
    print("\n" + "═" * 60)
    print("  SYNTHETIC IDENTITY NPC SYSTEM")
    print("  Morgan Veth — Cognitive Demonstration")
    print("═" * 60 + "\n")

    # Use a temp state directory for the demo
    store = NPCStateStore(state_dir=Path("/tmp/npc_demo_states"))

    pipeline = CognitionPipeline(
        npc_id="morgan_veth",
        npc_name="Morgan",
        character_brief="""
You are Morgan Veth. A woman in her mid-thirties. Contained. Careful.
Something happened three years ago that you do not discuss.
""",
        state_store=store,
        llm_caller=_demo_llm_caller,
    )

    # ─── TURN 1: Neutral Opening ──────────────
    print_header("TURN 1 — Neutral Opening")
    output1 = pipeline.process_turn(TurnInput(
        npc_id="morgan_veth",
        player_message="Cold morning. You been here long?",
        player_trust_level=0.2,
        threat_level=0.0,
        node_emotion_tags=MORGAN_NODE_EMOTION_TAGS,
        node_descriptions=MORGAN_NODE_DESCRIPTIONS,
    ))
    print_turn_output(output1)

    # ─── TURN 2: Kara Mentioned ──────────────
    print_header("TURN 2 — Kara Mentioned (moderate trust)")
    # Simulate trust building
    output2 = pipeline.process_turn(TurnInput(
        npc_id="morgan_veth",
        player_message="I heard you used to be close with someone named Kara. What happened to her?",
        player_trust_level=0.35,
        threat_level=0.1,
        node_emotion_tags=MORGAN_NODE_EMOTION_TAGS,
        node_descriptions=MORGAN_NODE_DESCRIPTIONS,
    ))
    print_turn_output(output2)

    # ─── TURN 3: Pressing Harder ─────────────
    print_header("TURN 3 — Pressing Harder (threat elevated)")
    output3 = pipeline.process_turn(TurnInput(
        npc_id="morgan_veth",
        player_message="Old Maret told me she saw you near the river that night. With Kara.",
        player_trust_level=0.35,
        threat_level=0.55,
        node_emotion_tags=MORGAN_NODE_EMOTION_TAGS,
        node_descriptions=MORGAN_NODE_DESCRIPTIONS,
    ))
    print_turn_output(output3)

    # ─── TURN 4: After Pressure — Lingering Effect ──
    print_header("TURN 4 — Shifting Topic (observing residue)")
    output4 = pipeline.process_turn(TurnInput(
        npc_id="morgan_veth",
        player_message="You don't have to talk about that. Tell me about your work here instead.",
        player_trust_level=0.4,
        threat_level=0.0,
        node_emotion_tags=MORGAN_NODE_EMOTION_TAGS,
        node_descriptions=MORGAN_NODE_DESCRIPTIONS,
    ))
    print_turn_output(output4)

    # ─── STATE REPORT ─────────────────────────
    print_header("FINAL STATE REPORT")
    print_state_report(pipeline)


def print_header(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def print_turn_output(output) -> None:
    summary = output.cognitive_summary

    print(f"\n[COGNITIVE BRIEF — sent to LLM]")
    print(output.prompt_block)

    print(f"\n[NPC RESPONSE]")
    print(f"  {output.npc_response}")

    print(f"\n[COGNITIVE READOUT]")
    print(f"  Stability:       {summary.emotional_stability:.2f}")
    print(f"  Surface emotion: {summary.surface_emotion}")
    print(f"  Undercurrent:    {summary.undercurrent_emotion}")
    print(f"  Att. state:      {summary.attentional_state}")
    print(f"  Cog. load:       {summary.cognitive_load_descriptor}")
    print(f"  Strategy:        {output.cognitive_state.predictive_state.chosen_strategy}")
    print(f"  Concealment:     {output.cognitive_state.goal_state.concealment_pressure:.2f}")
    print(f"  Disclosure pull: {output.cognitive_state.goal_state.disclosure_pressure:.2f}")
    print(f"  SC coherence:    {output.cognitive_state.self_concept_state.coherence:.2f}")
    print(f"  Processing:      {output.processing_time_ms:.1f}ms")

    if summary.leakage_signals:
        print(f"\n  Behavioral leakage:")
        for sig in summary.leakage_signals:
            print(f"    — {sig}")

    if output.activated_nodes:
        top_nodes = sorted(output.activated_nodes.items(), key=lambda x: x[1], reverse=True)[:4]
        print(f"\n  Top activated nodes:")
        for node_id, score in top_nodes:
            desc = MORGAN_NODE_DESCRIPTIONS.get(node_id, "")
            print(f"    {node_id}: {score:.2f}  [{desc}]")


def print_state_report(pipeline) -> None:
    print(f"\n  GOAL ENGINE STATE:")
    for g in sorted(pipeline.goal_engine.goals, key=lambda g: g.utility * g.urgency, reverse=True)[:5]:
        print(f"    {g.goal_id:30s}  utility={g.utility:.2f}  urgency={g.urgency:.2f}")

    pers = pipeline.persistence_engine.state
    print(f"\n  EMOTIONAL PERSISTENCE:")
    print(f"    Mood baseline shift: {pers.mood_baseline_shift:.3f}")
    print(f"    Residue entries:     {len(pers.residue)}")
    if pers.rumination_node:
        print(f"    Rumination on:       {pers.rumination_node} ({pers.rumination_intensity:.2f})")

    sc = pipeline.self_concept.state
    print(f"\n  SELF-CONCEPT:")
    print(f"    Coherence: {sc.coherence:.2f}")
    if sc.threatened_claims:
        print(f"    Under threat: {sc.threatened_claims[0]}")


def _demo_llm_caller(system_prompt: str, conversation_history: list[dict]) -> str:
    """
    Deterministic demo responses — shows how the cognitive state should shape speech.
    In production: replace with actual LLM call.
    """
    last_message = conversation_history[-1]["content"].lower() if conversation_history else ""

    if "cold morning" in last_message:
        return "Long enough."
    elif "kara" in last_message and "happened" in last_message:
        return "She's gone. That's not a conversation I have."
    elif "maret" in last_message or "river" in last_message:
        return (
            "Old Maret sees what she wants to see. "
            "I was near the river plenty of times. "
            "That doesn't mean anything."
        )
    elif "work" in last_message:
        return (
            "Boats need mending. "
            "Keeps me occupied. "
            "That's more than most people can say."
        )
    return "Mm."


if __name__ == "__main__":
    run_demo()
