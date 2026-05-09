def calculate_cognition_scores(profile, emotions):

    scores = {
        "Focus Score": 50,
        "Relaxation Score": 50,
        "Motivation Score": 50,
        "Emotional Stability": 50,
        "Cognitive Load": 50
    }

    # ---------------- FOCUS ----------------

    if profile["focus_suitability"] == "high":
        scores["Focus Score"] += 35

    elif profile["focus_suitability"] == "low":
        scores["Focus Score"] -= 25

    if profile["texture"] == "noisy":
        scores["Focus Score"] -= 15

    # ---------------- RELAXATION ----------------

    if profile["stimulation_level"] == "calm":
        scores["Relaxation Score"] += 35

    if profile["energy_level"] == "high":
        scores["Relaxation Score"] -= 25

    if profile["emotion"] == "calm":
        scores["Relaxation Score"] += 15

    # ---------------- MOTIVATION ----------------

    if profile["energy_level"] == "high":
        scores["Motivation Score"] += 30

    if profile["tempo_type"] == "fast":
        scores["Motivation Score"] += 20

    # ---------------- EMOTIONAL STABILITY ----------------

    transitions = 0

    for i in range(1, len(emotions)):
        if emotions[i] != emotions[i - 1]:
            transitions += 1

    if transitions <= 2:
        scores["Emotional Stability"] += 30

    elif transitions >= 6:
        scores["Emotional Stability"] -= 20

    # ---------------- COGNITIVE LOAD ----------------

    if profile["stimulation_level"] == "intense":
        scores["Cognitive Load"] += 35

    if profile["brightness"] == "bright":
        scores["Cognitive Load"] += 15

    if profile["texture"] == "noisy":
        scores["Cognitive Load"] += 15

    # ---------------- NORMALIZE ----------------

    for key in scores:
        scores[key] = max(0, min(100, scores[key]))

    return scores