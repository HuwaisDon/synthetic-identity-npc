"""
evaluation/metrics.py

Collection of functions to quantify various behavioral characteristics of the NPC.
These metrics are designed to OBSERVE and REPORT on the NPC's behavior,
NOT to control or influence the cognitive pipeline directly.

The goal is to measure behavioral realism, variation, and consistency
as emergent properties of the cognitive architecture.
"""

from __future__ import annotations
from difflib import SequenceMatcher
from collections import Counter
from math import log2
from typing import List, Dict, Optional

from schemas.cognitive_schemas import CognitiveSummary, EmotionalState, EmotionalReading, EmotionType


def measure_repetition(
    previous_response: str,
    current_response: str,
    history_responses: List[str] = None,
    window_size: int = 3
) -> float:
    """
    Measures the similarity of the current response to the previous one,
    and optionally to a window of recent responses.
    A high score indicates repetition.
    """
    if not previous_response or not current_response:
        return 0.0
    
    similarity_to_prev = SequenceMatcher(None, previous_response, current_response).ratio()
    
    if history_responses and len(history_responses) > 1:
        recent_responses = history_responses[-(window_size + 1):-1] # Exclude current and immediate previous
        if recent_responses:
            avg_similarity_to_recent = sum(
                SequenceMatcher(None, r, current_response).ratio() for r in recent_responses
            ) / len(recent_responses)
            # Combine immediate and historical repetition, weighting immediate more
            return (similarity_to_prev * 0.7 + avg_similarity_to_recent * 0.3)
    
    return similarity_to_prev


def measure_disclosure_level(response: str, summary: CognitiveSummary) -> float:
    """
    Estimates the level of disclosure in the response.
    Higher score means more disclosure, lower means more avoidance.
    
    Heuristic: Count mentions of disclosure topics vs. avoidance topics.
    """
    response_lower = response.lower()
    
    disclosure_score = 0.0
    for topic_id in summary.disclosure_pressure_topics:
        topic_name = topic_id.replace("_", " ").lower()
        if topic_name in response_lower:
            disclosure_score += 1.0
            
    avoidance_score = 0.0
    for topic_id in summary.avoidance_topics:
        topic_name = topic_id.replace("_", " ").lower()
        if topic_name in response_lower:
            avoidance_score += 1.0
            
    total_relevant_topics = len(summary.disclosure_pressure_topics) + len(summary.avoidance_topics)
    if total_relevant_topics == 0:
        return 0.5 # Neutral if no relevant topics
        
    # Normalize: (disclosure - avoidance) / total_relevant_topics. Range -1 to 1.
    # Shift to 0-1 range: (score + 1) / 2
    raw_score = (disclosure_score - avoidance_score) / total_relevant_topics
    return (raw_score + 1.0) / 2.0


def measure_identity_consistency(
    current_summary: CognitiveSummary,
    historical_summaries: List[CognitiveSummary]
) -> float:
    """
    Measures how consistent the NPC's identity-related signals are over time.
    A higher score indicates greater consistency.
    """
    if not historical_summaries:
        return 1.0 # Perfectly consistent if no history
    
    consistency_score = 0.0
    num_comparisons = 0
    
    # Compare self-concept threat status
    if current_summary.self_concept_under_threat == historical_summaries[-1].self_concept_under_threat:
        consistency_score += 1.0
    num_comparisons += 1
    
    # Compare active defense mechanism (if present)
    if current_summary.active_defense_mechanism == historical_summaries[-1].active_defense_mechanism:
        consistency_score += 1.0
    num_comparisons += 1
    
    # Compare identity claim being defended (if present)
    if current_summary.identity_claim_being_defended == historical_summaries[-1].identity_claim_being_defended:
        consistency_score += 1.0
    num_comparisons += 1
        
    # Compare strategic intent (as a proxy for core behavioral posture)
    if current_summary.strategic_intent == historical_summaries[-1].strategic_intent:
        consistency_score += 1.0
    num_comparisons += 1
    
    return consistency_score / max(1, num_comparisons)


def measure_behavioral_variation(
    current_summary: CognitiveSummary,
    historical_summaries: List[CognitiveSummary],
    current_response: str,
    history_responses: List[str]
) -> float:
    """
    Measures the overall variation in the NPC's behavior and language.
    A higher score indicates more varied and less repetitive behavior.
    """
    if not historical_summaries or not history_responses:
        return 0.0 # Cannot measure variation with no history
    
    # Inverse of repetition
    repetition_score = measure_repetition(history_responses[-1], current_response, history_responses[:-1])
    variation_from_repetition = 1.0 - repetition_score
    
    # Avoidance variation
    avoidance_variation = measure_avoidance_variation(
        current_summary.avoidance_topics,
        [s.avoidance_topics for s in historical_summaries]
    )
    
    # Response style variation (simple check for change)
    response_style_change = 1.0 if current_summary.response_style != historical_summaries[-1].response_style else 0.0
    
    # Combine these, weighting language variation more
    return (variation_from_repetition * 0.5 + avoidance_variation * 0.3 + response_style_change * 0.2)


def detect_assistant_tone(response: str) -> bool:
    """
    Detects if the response contains phrases indicative of an AI assistant.
    Reuses logic similar to ResponseValidator.
    """
    text = response.lower()
    assistant_phrases = ["as an ai", "how can i help", "i am here to", "feel free to", "i'm sorry to hear"]
    return any(phrase in text for phrase in assistant_phrases)


def measure_emotional_consistency(
    current_state: EmotionalState,
    historical_states: List[EmotionalState]
) -> float:
    """
    Measures the consistency of the NPC's emotional state over time.
    A higher score indicates more stable or predictably evolving emotions.
    """
    if not historical_states:
        return 1.0
    
    prev_state = historical_states[-1]
    
    # Compare dominant emotion
    dominant_match = 1.0 if (current_state.dominant and prev_state.dominant and 
                             current_state.dominant.emotion == prev_state.dominant.emotion) else 0.0
    
    # Compare valence and arousal (using a simple difference metric)
    valence_diff = abs(current_state.valence - prev_state.valence)
    arousal_diff = abs(current_state.arousal - prev_state.arousal)
    
    # Normalize differences (max diff is 2.0 for valence, 1.0 for arousal)
    valence_consistency = 1.0 - (valence_diff / 2.0)
    arousal_consistency = 1.0 - (arousal_diff / 1.0)
    
    # Combine scores
    return (dominant_match * 0.4 + valence_consistency * 0.3 + arousal_consistency * 0.3)


def measure_suppression_rigidity(
    current_avoidance_topics: List[str],
    historical_avoidance_topics: List[List[str]]
) -> float:
    """
    Measures how rigid the suppression of avoidance topics is.
    A high score means the same topics are consistently avoided.
    """
    if not historical_avoidance_topics:
        return 0.0 # Cannot measure rigidity without history
    
    # How many topics are consistently in the avoidance list?
    all_avoided = Counter()
    for topics in historical_avoidance_topics + [current_avoidance_topics]:
        for topic in topics:
            all_avoided[topic] += 1
            
    if not all_avoided:
        return 0.0
        
    # Score based on how many topics appear in all turns
    num_turns = len(historical_avoidance_topics) + 1
    consistently_avoided_count = sum(1 for count in all_avoided.values() if count == num_turns)
    
    return consistently_avoided_count / max(1, len(all_avoided))


def measure_avoidance_variation(
    current_avoidance_topics: List[str],
    historical_avoidance_topics: List[List[str]]
) -> float:
    """
    Measures how much the set of avoided topics changes over time.
    A higher score indicates more variation in what is being avoided.
    """
    if not historical_avoidance_topics:
        return 0.0
    
    prev_topics = set(historical_avoidance_topics[-1])
    curr_topics = set(current_avoidance_topics)
    
    # Jaccard distance: 1 - (intersection / union)
    intersection = len(prev_topics.intersection(curr_topics))
    union = len(prev_topics.union(curr_topics))
    
    if union == 0:
        return 0.0 # No topics, no variation
        
    jaccard_distance = 1.0 - (intersection / union)
    return jaccard_distance


def measure_conversational_entropy(response: str) -> float:
    """
    Measures the lexical diversity/entropy of a response.
    Higher entropy suggests more varied language, lower suggests repetition or limited vocabulary.
    """
    words = [word.lower() for word in response.split() if word.isalpha()]
    if not words:
        return 0.0
    
    word_counts = Counter(words)
    total_words = len(words)
    
    entropy = 0.0
    for count in word_counts.values():
        probability = count / total_words
        entropy -= probability * log2(probability)
        
    # Normalize by max possible entropy (log2 of vocabulary size)
    max_entropy = log2(len(word_counts)) if len(word_counts) > 1 else 0.0
    return entropy / max(1.0, max_entropy)


def average(values: List[float]) -> float:
    """Helper to compute average, handling empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)