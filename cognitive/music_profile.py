import numpy as np


def classify_tempo(tempo):
    if tempo < 90:
        return "slow"
    elif tempo < 130:
        return "medium"
    else:
        return "fast"


def classify_energy(energy):
    if energy < 0.03:
        return "low"
    elif energy < 0.08:
        return "moderate"
    else:
        return "high"


def classify_brightness(spectral_centroid):
    if spectral_centroid < 2000:
        return "dark"
    elif spectral_centroid < 4000:
        return "balanced"
    else:
        return "bright"


def classify_texture(zcr):
    if zcr < 0.05:
        return "smooth"
    elif zcr < 0.12:
        return "textured"
    else:
        return "noisy"


def classify_bandwidth(bandwidth):
    if bandwidth < 1500:
        return "narrow"
    elif bandwidth < 3000:
        return "moderate"
    else:
        return "wide"


def classify_stimulation(tempo_type, energy_level):
    if tempo_type == "fast" and energy_level == "high":
        return "intense"

    if tempo_type == "slow" and energy_level == "low":
        return "calm"

    return "balanced"


def classify_focus_suitability(stimulation, texture):
    if stimulation == "intense" or texture == "noisy":
        return "low"

    if stimulation == "calm" and texture == "smooth":
        return "high"

    return "moderate"


def build_music_profile(feature_dict, emotion, genre):

    tempo = feature_dict["tempo"]
    energy = feature_dict["energy"]
    spectral_centroid = feature_dict["spectral_centroid"]
    spectral_bandwidth = feature_dict["spectral_bandwidth"]
    zcr = feature_dict["zcr"]

    tempo_type = classify_tempo(tempo)
    energy_level = classify_energy(energy)
    brightness = classify_brightness(spectral_centroid)
    texture = classify_texture(zcr)
    bandwidth = classify_bandwidth(spectral_bandwidth)

    stimulation = classify_stimulation(
        tempo_type,
        energy_level
    )

    focus_suitability = classify_focus_suitability(
        stimulation,
        texture
    )

    profile = {
        "genre": genre,
        "emotion": emotion,

        "tempo_bpm": round(float(tempo), 2),
        "tempo_type": tempo_type,

        "energy_value": round(float(energy), 4),
        "energy_level": energy_level,

        "brightness": brightness,
        "texture": texture,
        "bandwidth": bandwidth,

        "stimulation_level": stimulation,
        "focus_suitability": focus_suitability
    }

    return profile