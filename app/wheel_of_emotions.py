WHEEL_OF_EMOTIONS: dict[str, dict[str, list[str]]] = {
    "happy": {
        "playful": ["aroused", "cheeky"],
        "content": ["free", "joyful"],
        "interested": ["curious", "inquisitive"],
        "proud": ["confident", "successful"],
        "accepted": ["valued", "respected"],
        "powerful": ["courageous", "creative"],
        "peaceful": ["loving", "thankful"],
        "trusting": ["sensitive", "intimate"],
        "optimistic": ["hopeful", "inspired"],
    },
    "sad": {
        "lonely": ["isolated", "abandoned"],
        "vulnerable": ["victimized", "fragile"],
        "despair": ["grief", "powerless"],
        "guilty": ["ashamed", "remorseful"],
        "depressed": ["empty", "inferior"],
        "hurt": ["disappointed", "embarrassed"],
    },
    "disgusted": {
        "disapproving": ["judgmental", "embarrassed"],
        "disappointed": ["appalled", "revolted"],
        "awful": ["nauseated", "detestable"],
        "repelled": ["horrified", "hesitant"],
    },
    "angry": {
        "let down": ["betrayed", "resentful"],
        "humiliated": ["disrespected", "ridiculed"],
        "bitter": ["indignant", "violated"],
        "mad": ["furious", "jealous"],
        "aggressive": ["hostile", "provoked"],
        "frustrated": ["annoyed", "infuriated"],
        "distant": ["withdrawn", "numb"],
        "critical": ["skeptical", "dismissive"],
    },
    "surprised": {
        "startled": ["shocked", "dismayed"],
        "confused": ["perplexed", "disillusioned"],
        "amazed": ["astonished", "awe"],
        "excited": ["eager", "energetic"],
    },
    "fearful": {
        "scared": ["helpless", "frightened"],
        "anxious": ["worried", "overwhelmed"],
        "insecure": ["inadequate", "inferior"],
        "weak": ["worthless", "insignificant"],
        "rejected": ["excluded", "persecuted"],
        "threatened": ["nervous", "exposed"],
    },
    "bad": {
        "bored": ["indifferent", "apathetic"],
        "busy": ["pressured", "rushed"],
        "stressed": ["overwhelmed", "out of control"],
        "tired": ["sleepy", "unfocused"],
    },
}


def get_wheel_of_emotions():
    return WHEEL_OF_EMOTIONS


def get_emotion_depth(mood: str, wheel: dict) -> int:
    """Determine the depth level of an emotion in the wheel (1=primary, 2=secondary, 3=tertiary)"""
    if not mood:
        return 0

    mood_lower = mood.lower()

    # Check primary level (top-level keys)
    if mood_lower in wheel:
        return 1

    # Check secondary level (second-level keys)
    for primary_mood, secondary_moods in wheel.items():
        if isinstance(secondary_moods, dict):
            if mood_lower in secondary_moods:
                return 2
            # Check tertiary level (third-level values)
            for secondary_mood, tertiary_moods in secondary_moods.items():
                if isinstance(tertiary_moods, list) and mood_lower in [
                    m.lower() for m in tertiary_moods
                ]:
                    return 3

    return 0  # Not found or unknown
