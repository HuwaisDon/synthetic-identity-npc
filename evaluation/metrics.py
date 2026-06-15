"""
evaluation/metrics.py

Lightweight evaluation utilities for Synthetic Identity NPCs.
Provides heuristics for measuring behavioral quality and cognitive alignment.
"""

from __future__ import annotations
from difflib import SequenceMatcher


def measure_repetition(responses: list[str]) -> float:
    """
    Measures lexical similarity between the current response and the previous one.
    
    Why it matters:
    Repetition harms realism because it exposes the underlying state machine. 
    Natural conversation uses varied syntax even when expressing the same idea.
    """
    if len(responses) < 2:
        return 0.0
    current = responses[-1]
    prev = responses[-2]
    return SequenceMatcher(None, current, prev).ratio()

def measure_disclosure_level(response: str) -> float:
    """
    Estimates the degree of revelation vs. deflection.
    
    Why it matters:
    Disclosure pacing matters because trust should be an emergent property of 
    interaction. Immediate 'data dumping' of secrets breaks psychological immersion.
    """
    text = response.lower()
    # Markers of avoidance or deflection
    avoidance_markers = ["not now", "don't ask", "none of your", "another time", "anyway", "forget it"]
    # Markers of specific detail or vulnerability
    revelation_markers = ["because", "actually", "i remember", "the truth", "it was", "happened"]
    
    score = 0.4
    for m in avoidance_markers:
        if m in text: score -= 0.15
    for m in revelation_markers:
        if m in text: score += 0.1
        
    return max(0.0, min(1.0, score))

def measure_behavioral_variation(responses: list[str]) -> float:
    """
    Measures how much the NPC varies its output over time.
    """
    if not responses:
        return 1.0
    return 1.0 - measure_repetition(responses)

def detect_assistant_tone(response: str) -> bool:
    """
    Detects linguistic cues associated with standard AI assistant helpfulness.
    
    Why it matters:
    Assistant-tone detection matters because it signals that the persona has 
    collapsed into a generic AI pattern, losing character specificity.
    """
    markers = ["as an ai", "how can i help", "i'm here to assist", "i apologize", "feel free to"]
    text = response.lower()
    return any(m in text for m in markers)

def measure_identity_consistency(responses: list[str]) -> float:
    """
    Measures the stability of the persona across turns by checking structural consistency.
    """
    if len(responses) < 3:
        return 1.0
    lengths = [len(r.split()) for r in responses]
    avg_len = sum(lengths) / len(lengths)
    # Excessive length variance often indicates 'tone collapse'
    variance = sum(abs(l - avg_len) for l in lengths) / (avg_len * len(lengths))
    return max(0.0, min(1.0, 1.0 - variance))