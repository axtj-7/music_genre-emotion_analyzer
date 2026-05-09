from collections import Counter
import numpy as np


def analyze_emotion_transitions(emotions):

    unique_emotions = len(set(emotions))

    if unique_emotions == 1:
        return (
            "The emotional profile remains highly consistent throughout the track, "
            "indicating stable emotional pacing."
        )

    transitions = 0

    for i in range(1, len(emotions)):
        if emotions[i] != emotions[i - 1]:
            transitions += 1

    if transitions <= 2:
        return (
            "The track demonstrates mild emotional variation with relatively stable transitions."
        )

    elif transitions <= 5:
        return (
            "The emotional progression varies noticeably across segments, "
            "creating moderate emotional dynamism."
        )

    else:
        return (
            "The emotional profile fluctuates significantly across the track, "
            "indicating inconsistent emotional pacing and high emotional variability."
        )


def generate_behavioral_analysis(profile, emotions):

    lines = []

    tempo = profile["tempo_bpm"]
    energy = profile["energy_level"]
    brightness = profile["brightness"]
    stimulation = profile["stimulation_level"]
    emotion = profile["emotion"]

    # ---------------- TEMPO ----------------

    if tempo >= 130:
        lines.append(
            f"The track contains a fast rhythmic structure (~{tempo:.0f} BPM), "
            "which increases cognitive stimulation and perceived intensity."
        )

    elif tempo <= 90:
        lines.append(
            f"The music maintains a slow tempo profile (~{tempo:.0f} BPM), "
            "supporting calmer cognitive engagement."
        )

    # ---------------- ENERGY ----------------

    if energy == "high":
        lines.append(
            "The audio exhibits elevated energy characteristics, "
            "suggesting strong motivational and stimulation patterns."
        )

    elif energy == "low":
        lines.append(
            "The music demonstrates low-energy acoustic behavior, "
            "contributing to a softer and less intrusive listening experience."
        )

    # ---------------- BRIGHTNESS ----------------

    if brightness == "bright":
        lines.append(
            "High spectral brightness indicates intense high-frequency activity "
            "and sharper acoustic texture."
        )

    elif brightness == "dark":
        lines.append(
            "The darker spectral profile creates a warmer and smoother auditory perception."
        )

    # ---------------- EMOTION ----------------

    if emotion == "calm":
        lines.append(
            "The detected emotional tone remains emotionally stable and relaxation-oriented."
        )

    elif emotion == "angry":
        lines.append(
            "Aggressive emotional characteristics introduce heightened emotional intensity."
        )

    elif emotion == "happy":
        lines.append(
            "Positive emotional patterns contribute to an uplifting auditory atmosphere."
        )

    # ---------------- STIMULATION ----------------

    if stimulation == "intense":
        lines.append(
            "Combined rhythmic and spectral patterns indicate strong sensory stimulation."
        )

    elif stimulation == "calm":
        lines.append(
            "The stimulation profile remains controlled and cognitively lightweight."
        )

    # ---------------- EMOTION TIMELINE ----------------

    transition_analysis = analyze_emotion_transitions(emotions)

    lines.append(transition_analysis)

    return lines