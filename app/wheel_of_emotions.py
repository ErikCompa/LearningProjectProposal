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
