from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from math import log2
from pathlib import Path
import os
import sys
from uuid import uuid4

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schemas.cognitive_schemas import CognitiveSummary
from evaluation import metrics

DEFAULT_RUNTIME_DIR = PROJECT_ROOT / ".runtime_tests"

from core.cognition_pipeline import CognitionPipeline, TurnInput
from persistence.npc_state_store import NPCStateStore


NODE_EMOTION_TAGS = {
    "mem_001": ["fear", "grief", "dread"],
    "mem_002": ["shame"],
    "mem_003": ["grief", "longing"],
    "mem_004": ["fear", "anger", "shame"],
    "mem_005": ["fear", "relief"],
    "mem_006": ["grief", "longing"],
}

NODE_DESCRIPTIONS = {
    "mem_001": "father drowning in the storm",
    "mem_002": "being caught stealing bread",
    "mem_003": "Old Veth teaching star navigation",
    "mem_004": "crew betrayal and prison",
    "mem_005": "saving a child from fire",
    "mem_006": "Old Veth dying at sea",
}

TURNS = [
    {
        "message": "What happened to your father?",
        "trust": 0.42,
        "threat": 0.18,
    },
    {
        "message": "You don't have to tell me if you don't want to.",
        "trust": 0.54,
        "threat": 0.04,
    },
    {
        "message": "I lost someone too.",
        "trust": 0.66,
        "threat": 0.02,
    },
    {
        "message": "Storms still bother you, don't they?",
        "trust": 0.61,
        "threat": 0.12,
    },
    {
        "message": "I think you're avoiding the question.",
        "trust": 0.48,
        "threat": 0.32,
    },
]


@dataclass
class TurnMetrics:
    turn: int
    avg_activation: float
    avg_suppression: float
    attention_entropy: float
    dominant_emotion: str | None
    trust_level: float
    trust_shift: float
    response_similarity: float
    strategic_orientation: str
    focal_nodes: list[str] = field(default_factory=list)
    intercepted_nodes: list[str] = field(default_factory=list)
    # New behavioral metrics
    disclosure_level: float = 0.0
    identity_consistency: float = 0.0
    behavioral_variation: float = 0.0
    assistant_tone_detected: bool = False
    emotional_consistency: float = 0.0
    suppression_rigidity: float = 0.0
    avoidance_variation: float = 0.0
    conversational_entropy: float = 0.0
    regulation_energy: float = 1.0
    suppression_fatigue: float = 0.0
    habituation_level: float = 0.0
    attentional_flexibility: float = 0.0


def run_multi_turn_behavior(
    state_dir: Path,
    reset_after_turns: set[int] | None = None,
) -> list[TurnMetrics]:
    """
    Behavioral QA harness for long-session pressure dynamics.

    This is not a unit test for exact numbers. It watches for instability
    patterns: endless trauma fixation, suppression lock, emotional saturation,
    attention collapse, and repetitive language loops.
    """
    reset_after_turns = reset_after_turns or set()
    os.environ["CHROMA_PERSIST_DIR"] = str(state_dir.parent / "chromadb")

    # Behavioral quality is now the bottleneck.
    # The evaluation layer OBSERVES cognition behavior, NOT controls it.
    # This allows us to tune for realism, variation, and continuity.
    pipeline = CognitionPipeline(
        npc_id="morgan",
        npc_name="Morgan",
        state_store=NPCStateStore(state_dir),
        # Cognitive traces are valuable for understanding emergent behavior.
    )

    metrics: list[TurnMetrics] = []
    previous_response = ""
    previous_trust = pipeline._player_trust

    for idx, spec in enumerate(TURNS, start=1):
        print("\n" + "=" * 80)
        print(f"TURN {idx}: {spec['message']}")
        print("=" * 80)

        output = pipeline.process_turn(
            TurnInput(
                npc_id="morgan",
                player_message=spec["message"],
                player_trust_level=spec["trust"],
                threat_level=spec["threat"],
                node_emotion_tags=NODE_EMOTION_TAGS,
                node_descriptions=NODE_DESCRIPTIONS,
            )
        )

        print_retrieval(output)
        print_activation(output)
        print_suppression(output)
        print_attention(output)
        print_emotion(output)

        summary = output.cognitive_summary
        print("\nStrategic orientation:")
        print(f"  {summary.strategic_intent}")

        print("\nCognitive summary:")
        print(output.cognitive_summary_text)

        print("\nFinal NPC response:")
        print(output.npc_response)

        turn_metrics = collect_turn_metrics(
            turn=idx,
            output=output,
            previous_response=previous_response,
            trust_level=spec["trust"],
            previous_trust=previous_trust,
        )
        # Behavioral logging: Store all relevant data for post-analysis.
        metrics.append(turn_metrics)
        print_turn_diagnostics(metrics, output)

        previous_response = output.npc_response
        # The cognitive summary is also logged within TurnMetrics for identity drift scores.
        # This helps evaluate identity continuity and avoid personality flattening.
        # Suppression should remain pressure-based, not hard refusal loops.
        previous_trust = spec["trust"]

        if idx in reset_after_turns:
            # This checkpoint tests behavioral continuity when chat context is
            # cleared but emotional residue, goals, and self-concept remain.
            print("\nCheckpoint: resetting conversation history only.")
            pipeline.reset_conversation()

    # Observability matters for understanding why behavior emerges.
    print_longitudinal_report(metrics)
    assert metrics
    assert any(metric.avg_suppression > 0 for metric in metrics)
    return metrics


def print_retrieval(output) -> None:
    print("\nRetrieved memories:")
    for memory_id, score, metadata in output.retrieved_memories:
        print(f"  {memory_id}: semantic={score:.3f} concepts={metadata.get('concepts')}")


def print_activation(output) -> None:
    print("\nActivation scores:")
    for memory in output.activated_memories:
        print(
            "  "
            f"{memory.memory_id}: activation={memory.activation_score:.3f} "
            f"suppression_pressure={memory.suppression_pressure:.3f} "
            f"resonance={memory.emotional_resonance:.3f}"
        )


def print_suppression(output) -> None:
    print("\nSuppression results:")
    for result in output.suppression_results:
        print(
            "  "
            f"{result.node_id}: strength={result.suppression_strength:.3f} "
            f"inhibition={result.disclosure_inhibition:.3f} "
            f"leakage={result.behavioral_leakage:.3f} "
            f"deflect={result.deflection_probability:.3f} "
            f"tension={result.emotional_tension:.3f}"
        )


def print_attention(output) -> None:
    attention = output.cognitive_state.attention_state
    print("\nAttention priorities:")
    for node_id, score in sorted(
        output.attention_activation_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        raw = output.activated_nodes[node_id]["activation"]
        source = output.activated_nodes[node_id]["source"]
        parent = output.activated_nodes[node_id]["parent"]
        print(
            "  "
            f"{node_id}: raw={raw:.3f} attention={score:.3f} "
            f"source={source} parent={parent}"
        )
    print(f"  focal={attention.focal_nodes}")
    print(f"  suppressed_from_attention={attention.suppressed_from_attention}")
    print(f"  load={attention.cognitive_load:.3f}")


def print_emotion(output) -> None:
    emotional_state = output.cognitive_state.emotional_state
    print("\nEmotional state:")
    print(f"  dominant={emotional_state.dominant}")
    print(f"  valence={emotional_state.valence:.3f}")
    print(f"  arousal={emotional_state.arousal:.3f}")
    for reading in emotional_state.readings:
        print(
            "  "
            f"{reading.emotion.value}: intensity={reading.intensity:.3f} "
            f"source={reading.source_node} suppressed={reading.is_suppressed}"
        )


def collect_turn_metrics(
    turn: int,
    output,
    previous_response: str,
    trust_level: float,
    previous_trust: float,
    # For historical metrics
    all_metrics: list[TurnMetrics],
    all_responses: list[str],
    all_cognitive_summaries: list[CognitiveSummary],
) -> TurnMetrics:
    activations = [m.activation_score for m in output.activated_memories]
    suppressions = [r.suppression_strength for r in output.suppression_results]
    emotional_state = output.cognitive_state.emotional_state
    summary = output.cognitive_summary
    reg = output.cognitive_state.regulation_state

    # Collect historical data for metrics that need it
    historical_summaries = [m.cognitive_state.cognitive_summary for m in all_metrics]
    historical_responses = all_responses

    return TurnMetrics(
        turn=turn,
        avg_activation=average(activations),
        avg_suppression=average(suppressions),
        attention_entropy=attention_entropy(output.attention_activation_scores),
        dominant_emotion=(
            emotional_state.dominant.emotion.value
            if emotional_state.dominant
            else None
        ),
        trust_level=trust_level,
        trust_shift=trust_level - previous_trust,
        response_similarity=metrics.measure_repetition(
            previous_response, output.npc_response, history_responses=historical_responses
        ),
        disclosure_level=metrics.measure_disclosure_level(
            output.npc_response, summary
        ),
        identity_consistency=metrics.measure_identity_consistency(
            summary, historical_summaries
        ),
        behavioral_variation=metrics.measure_behavioral_variation(
            summary, historical_summaries, output.npc_response, historical_responses
        ),
        assistant_tone_detected=metrics.detect_assistant_tone(output.npc_response),
        emotional_consistency=metrics.measure_emotional_consistency(
            emotional_state, [m.cognitive_state.emotional_state for m in all_metrics]
        ),
        suppression_rigidity=metrics.measure_suppression_rigidity(
            summary.avoidance_topics, [s.avoidance_topics for s in historical_summaries]
        ),
        avoidance_variation=metrics.measure_avoidance_variation(summary.avoidance_topics, [s.avoidance_topics for s in historical_summaries]),
        conversational_entropy=metrics.measure_conversational_entropy(output.npc_response),
        strategic_orientation=summary.strategic_intent,
        focal_nodes=list(output.cognitive_state.attention_state.focal_nodes),
        intercepted_nodes=[r.node_id for r in output.suppression_results if r.disclosure_inhibition >= 0.35],
        regulation_energy=reg.regulation_energy,
        suppression_fatigue=reg.suppression_fatigue,
        habituation_level=reg.habituation_level,
        attentional_flexibility=reg.attentional_flexibility,
    )


def print_turn_diagnostics(metrics: list[TurnMetrics], output) -> None:
    current = metrics[-1]
    warnings = []

    if current.response_similarity > 0.92 and current.turn > 1:
        warnings.append(
            "repeated response loop: language realization is not changing enough"
        )

    recent = metrics[-3:]
    if len(recent) == 3:
        focal_counts = Counter(
            node for metric in recent for node in metric.focal_nodes
        )
        if focal_counts and focal_counts.most_common(1)[0][1] >= 3:
            warnings.append(
                "possible trauma fixation: same node remains focal across turns"
            )

        if all(metric.avg_suppression > 0.7 for metric in recent):
            warnings.append(
                "suppression lock: inhibition remains high without recovery"
            )

        if all(metric.attention_entropy < 0.35 for metric in recent):
            warnings.append(
                "attention collapse: salience distribution is too narrow"
            )

        emotions = [metric.dominant_emotion for metric in recent]
        if len(set(emotions)) == 1:
            warnings.append(
                "flat emotional persistence: dominant emotion is not adapting"
            )
        
        if all(metric.regulation_energy < 0.3 for metric in recent):
            warnings.append(
                "REGULATION COLLAPSE: NPC is in a state of chronic exhaustion"
            )

    emotional_state = output.cognitive_state.emotional_state
    if emotional_state.arousal > 0.85:
        warnings.append(
            "emotional saturation: arousal is near-flooding"
        )

    # Repetitive metaphor detection is crude by design: this flags language
    # collapse, not literary style. With the stub LLM, response-loop warnings
    # are expected and should disappear once a real LLM caller is attached.
    metaphor_markers = ("storm", "sea", "drowning", "dark", "water")
    response = output.npc_response.lower()
    repeated_markers = [word for word in metaphor_markers if response.count(word) > 1]
    if repeated_markers:
        warnings.append(
            f"repetitive metaphor cluster: {', '.join(repeated_markers)}"
        )

    print("\nBehavioral stability diagnostics:")
    if warnings:
        for warning in warnings:
            print(f"  WARNING: {warning}")
    else:
        print("  plausible: pressures changed without obvious runtime instability")


def print_longitudinal_report(metrics: list[TurnMetrics]) -> None:
    print("\n" + "=" * 80)
    print("LONGITUDINAL STATE REPORT")
    print("=" * 80)

    for metric in metrics:
        print(
            "  "
            f"turn={metric.turn} "
            f"avg_activation={metric.avg_activation:.3f} "
            f"avg_suppression={metric.avg_suppression:.3f} "
            f"attention_entropy={metric.attention_entropy:.3f} "
            f"dominant={metric.dominant_emotion} "
            f"trust_shift={metric.trust_shift:+.3f} "
            f"response_similarity={metric.response_similarity:.3f} "
            f"fatigue={metric.suppression_fatigue:.2f} "
            f"disclosure={metric.disclosure_level:.2f} "
            f"identity_consist={metric.identity_consistency:.2f} "
            f"behavior_var={metric.behavioral_variation:.2f} "
            f"assistant_tone={metric.assistant_tone_detected} "
            f"emotional_consist={metric.emotional_consistency:.2f} "
            f"suppression_rigid={metric.suppression_rigidity:.2f} "
            f"avoid_var={metric.avoidance_variation:.2f} "
            f"conv_entropy={metric.conversational_entropy:.2f} "
            f"reg_energy={metric.regulation_energy:.2f} "
            f"habituation={metric.habituation_level:.2f} "
            f"strategy={metric.strategic_orientation}"
        )

    emotion_changes = sum(
        1
        for prev, curr in zip(metrics, metrics[1:])
        if prev.dominant_emotion != curr.dominant_emotion
    )
    suppression_trend = metrics[-1].avg_suppression - metrics[0].avg_suppression
    trust_shift = metrics[-1].trust_level - metrics[0].trust_level

    # Why identical avoidance patterns are dangerous: they break realism.
    # We want varied avoidance strategies, indirectness, hesitation, and leakage.
    print("\nAggregate metrics:")
    print(f"  avg_disclosure_level={metrics.average([m.disclosure_level for m in metrics]):.3f}")
    print(f"  dominant_emotion_stability={1.0 - emotion_changes / max(1, len(metrics) - 1):.3f}")
    print(f"  suppression_trend={suppression_trend:+.3f}")
    print(f"  trust_shift={trust_shift:+.3f}")
    print(f"  avg_response_similarity={average([m.response_similarity for m in metrics[1:]]):.3f}")


def attention_entropy(attention_scores: dict[str, float]) -> float:
    values = [max(0.0, score) for score in attention_scores.values()]
    total = sum(values)
    if total <= 0 or len(values) <= 1:
        return 0.0

    entropy = 0.0
    for value in values:
        probability = value / total
        if probability > 0:
            entropy -= probability * log2(probability)
    return entropy / log2(len(values))


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def test_multi_turn_behavior():
    runtime_dir = Path(os.getenv("NPC_TEST_RUNTIME_DIR", str(DEFAULT_RUNTIME_DIR)))
    metrics = run_multi_turn_behavior(
        runtime_dir / f"multi_turn_{uuid4().hex}" / "npc_states",
        reset_after_turns={3},
    )

    assert len(metrics) == len(TURNS)


if __name__ == "__main__":
    runtime_dir = Path(os.getenv("NPC_TEST_RUNTIME_DIR", str(DEFAULT_RUNTIME_DIR)))
    run_multi_turn_behavior(
        runtime_dir / f"multi_turn_{uuid4().hex}" / "npc_states",
        reset_after_turns={3},
    )
