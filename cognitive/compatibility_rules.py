def evaluate_context(profile, user_mood, activity, goal):

    compatibility_score = 50
    reasons = []

    # ---------------- STUDYING ----------------

    if activity == "studying":

        if profile["stimulation_level"] == "intense":
            compatibility_score -= 25
            reasons.append(
                "High stimulation may reduce concentration during studying."
            )

        if profile["focus_suitability"] == "high":
            compatibility_score += 25
            reasons.append(
                "The music has characteristics suitable for focused tasks."
            )

        if profile["texture"] == "noisy":
            compatibility_score -= 15
            reasons.append(
                "Noisy audio texture may distract cognitive attention."
            )

        if profile["emotion"] == "calm":
            compatibility_score += 15
            reasons.append(
                "Calm emotional tone supports sustained mental focus."
            )

    # ---------------- SLEEP ----------------

    elif activity == "sleeping":

        if profile["energy_level"] == "high":
            compatibility_score -= 30
            reasons.append(
                "High energy music is not ideal for sleep preparation."
            )

        if profile["tempo_type"] == "slow":
            compatibility_score += 20
            reasons.append(
                "Slow tempo supports relaxation and reduced stimulation."
            )

        if profile["emotion"] == "calm":
            compatibility_score += 20
            reasons.append(
                "Calm emotional patterns are sleep-friendly."
            )

    # ---------------- GYM ----------------

    elif activity == "gym":

        if profile["energy_level"] == "high":
            compatibility_score += 25
            reasons.append(
                "High energy levels are suitable for workouts."
            )

        if profile["tempo_type"] == "fast":
            compatibility_score += 20
            reasons.append(
                "Fast tempo enhances movement intensity and motivation."
            )

        if profile["stimulation_level"] == "intense":
            compatibility_score += 15
            reasons.append(
                "Intense stimulation matches workout environments."
            )

    # ---------------- MEDITATION ----------------

    elif activity == "meditation":

        if profile["tempo_type"] == "slow":
            compatibility_score += 20
            reasons.append(
                "Slow tempo supports meditative environments."
            )

        if profile["energy_level"] == "low":
            compatibility_score += 20
            reasons.append(
                "Low energy music reduces mental overstimulation."
            )

        if profile["texture"] == "smooth":
            compatibility_score += 15
            reasons.append(
                "Smooth texture creates a calming atmosphere."
            )

    # ---------------- USER MOOD ----------------

    if user_mood == "stressed":

        if profile["emotion"] == "calm":
            compatibility_score += 20
            reasons.append(
                "Calm emotional tone may help reduce stress."
            )

        if profile["stimulation_level"] == "intense":
            compatibility_score -= 15
            reasons.append(
                "Highly intense music may increase mental stress."
            )

    elif user_mood == "sad":

        if profile["emotion"] == "happy":
            compatibility_score += 15
            reasons.append(
                "Positive emotional tone may improve mood."
            )

    elif user_mood == "motivated":

        if profile["energy_level"] == "high":
            compatibility_score += 15
            reasons.append(
                "High energy complements motivational states."
            )

    # ---------------- GOAL ----------------

    if goal == "improve focus":

        if profile["focus_suitability"] == "high":
            compatibility_score += 20
            reasons.append(
                "The music profile supports concentration and focus."
            )

        else:
            compatibility_score -= 10
            reasons.append(
                "The music may not optimally support deep focus."
            )

    elif goal == "relax":

        if profile["stimulation_level"] == "calm":
            compatibility_score += 20
            reasons.append(
                "The music provides a calming stimulation profile."
            )

    elif goal == "boost energy":

        if profile["energy_level"] == "high":
            compatibility_score += 20
            reasons.append(
                "The track contains strong energetic characteristics."
            )

    # ---------------- FINAL LIMITS ----------------

    compatibility_score = max(0, min(100, compatibility_score))

    # ---------------- FINAL LABEL ----------------

    if compatibility_score >= 80:
        alignment = "Highly Compatible"

    elif compatibility_score >= 60:
        alignment = "Moderately Compatible"

    elif compatibility_score >= 40:
        alignment = "Neutral"

    else:
        alignment = "Poorly Compatible"

    return {
        "score": compatibility_score,
        "alignment": alignment,
        "reasons": reasons
    }


# ---------------- TEST ----------------

if __name__ == "__main__":

    sample_profile = {
        "genre": "rock",
        "emotion": "angry",
        "tempo_type": "fast",
        "energy_level": "high",
        "brightness": "bright",
        "texture": "noisy",
        "bandwidth": "wide",
        "stimulation_level": "intense",
        "focus_suitability": "low"
    }

    result = evaluate_context(
        profile=sample_profile,
        user_mood="stressed",
        activity="studying",
        goal="improve focus"
    )

    print("\nCONTEXT ANALYSIS\n")

    print("Score:", result["score"])
    print("Alignment:", result["alignment"])

    print("\nReasons:")
    for r in result["reasons"]:
        print("-", r)