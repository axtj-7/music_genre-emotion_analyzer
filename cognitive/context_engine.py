from .music_profile import build_music_profile
from .compatibility_rules import evaluate_context
from .explain import generate_behavioral_analysis
from .scores import calculate_cognition_scores
def generate_explanation(result):

    score = result["score"]

    if score >= 80:
        summary = (
            "The selected music is highly suitable for the user's current "
            "emotional and cognitive context."
        )

    elif score >= 60:
        summary = (
            "The music is reasonably compatible with the user's current state, "
            "with minor conflicting characteristics."
        )

    elif score >= 40:
        summary = (
            "The music demonstrates mixed compatibility with the current context."
        )

    else:
        summary = (
            "The selected music may conflict with the user's emotional or "
            "cognitive requirements."
        )

    return summary


def generate_recommendation(result):

    score = result["score"]

    if score >= 80:
        return (
            "Recommended for the selected activity and emotional state."
        )

    elif score >= 60:
        return (
            "Usable in moderation depending on user preference."
        )

    elif score >= 40:
        return (
            "Consider switching to a calmer or more context-appropriate track."
        )

    else:
        return (
            "This track may not be psychologically suitable for the selected context."
        )


def analyze_music_context(
        feature_dict,
        emotions,
        emotion,
        genre,
        user_mood,
        activity,
        goal
):

    # ---------------- BUILD SEMANTIC PROFILE ----------------

    profile = build_music_profile(
        feature_dict=feature_dict,
        emotion=emotion,
        genre=genre
    )

    # ---------------- APPLY CONTEXT RULES ----------------

    compatibility_result = evaluate_context(
        profile=profile,
        user_mood=user_mood,
        activity=activity,
        goal=goal
    )

    # ---------------- GENERATE OUTPUTS ----------------

    explanation = generate_explanation(
        compatibility_result
    )

    recommendation = generate_recommendation(
        compatibility_result
    )

    behavioral_analysis = generate_behavioral_analysis(
        profile=profile,
        emotions=emotions
    )

    cognition_scores = calculate_cognition_scores(profile, emotions)

    # ---------------- FINAL RESPONSE ----------------

    final_output = {
        "music_profile": profile,

        "compatibility_score":
            compatibility_result["score"],

        "alignment":
            compatibility_result["alignment"],

        "reasons":
            compatibility_result["reasons"],

        "summary":
            explanation,

        "recommendation":
            recommendation,
        
        "behavioral_analysis": 
            behavioral_analysis,

        "cognition_scores":
            cognition_scores
    }

    return final_output


