import json

from fastapi import HTTPException

from app.deps import get_gemini_client
from app.wheel_of_emotions import get_wheel_of_emotions


async def analyze_mood(
    qa_pairs: list[tuple[str, str]],
    moods: list[tuple[str, float]],
    question: str,
    answer: str,
) -> tuple[str, float]:
    mood = ""
    mood_confidence = 0.0

    prompt = """
        You are an expert in emotional analysis with the wheel of emotions framework.

        Given the past user responses and your questions, as well as the latest question and answer pair,
        determine the user's current emotional state in terms of mood and confidence level.
        
        IMPORTANT: The wheel of emotions has 3 levels of depth:
        - Level 1 (Primary): happy, sad, angry, fearful, surprised, disgusted, bad
        - Level 2 (Secondary): playful, content, interested, proud, lonely, vulnerable, despair, guilty, etc.
        - Level 3 (Tertiary): aroused, cheeky, free, joyful, curious, isolated, abandoned, victimized, etc.
        
        Your goal is to identify the MOST SPECIFIC emotion possible from the user's responses.
        If you can determine a tertiary (level 3) emotion with confidence, use that.
        If only secondary (level 2) is clear, use that.
        Only fall back to primary (level 1) if the response is too vague.

        Work through the following steps:
        1. Review the previous question and answer pairs to understand the context.
        2. Analyze the latest answer in relation to the latest question.
        3. Map the emotional cues from the previous and current answers to the wheel of emotions.
        4. Determine the DEEPEST/MOST SPECIFIC fitting mood from the wheel of emotions (prefer level 3 > level 2 > level 1).
        5. Set confidence based on how clearly the emotion is expressed (specific details = higher confidence).
        6. Provide the mood and confidence score in the specified response format.
        
        WHEEL OF EMOTIONS:
        {wheel_of_emotions}

        USER QUESTION AND ANSWER HISTORY:
        {qa_history}

        USER LATEST QUESTION:
        {latest_question}

        USER LATEST ANSWER:
        {latest_answer}

        RESPONSE FORMAT:
        {{
            "mood": "<detected mood from the wheel of emotions - USE THE MOST SPECIFIC LEVEL POSSIBLE>",
            "confidence": <confidence score between 0 and 1>
        }}
    """

    prompt_filled = prompt.format(
        wheel_of_emotions=get_wheel_of_emotions(),
        qa_history="\n".join([f"Q: {q}\nA: {a}" for q, a in qa_pairs]),
        latest_question=question,
        latest_answer=answer,
    )
    print(prompt_filled)

    try:
        response = get_gemini_client().models.generate_content(
            model="gemini-3-pro-preview",
            contents=[prompt_filled],
            config={
                "response_mime_type": "application/json",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Mood analysis failed: {e}")

    try:
        result = json.loads(response.text)
        mood = result.get("mood", "")
        mood_confidence = result.get("confidence", 0.0)
    except (json.JSONDecodeError, KeyError) as e:
        raise HTTPException(
            status_code=400, detail=f"Mood analysis parsing failed: {e}"
        )

    return mood, mood_confidence


async def get_next_question(
    qa_pairs: list[tuple[str, str]],
    moods: list[tuple[str, float]],
    current_depth: int,
    max_depth: int,
) -> str:
    next_question = ""

    # Determine depth guidance for the LLM
    depth_names = ["unknown", "primary", "secondary", "tertiary"]
    current_depth_name = depth_names[current_depth] if current_depth <= 3 else "unknown"
    target_depth_name = (
        depth_names[min(current_depth + 1, max_depth)]
        if current_depth < max_depth
        else depth_names[max_depth]
    )

    prompt = """
        You are an expert in emotional analysis and questioning techniques using the wheel of emotions framework.

        Based on the previous question and answer pairs, as well as the detected moods and their confidence levels,
        generate the next most effective question to better understand the user's emotional state.

        CURRENT EMOTIONAL DEPTH: {current_depth_name} level (depth {current_depth})
        TARGET DEPTH: {target_depth_name} level (depth {target_depth})
        
        The wheel of emotions has 3 levels of depth:
        - Level 1 (Primary): happy, sad, angry, fearful, surprised, disgusted, bad
        - Level 2 (Secondary): playful, content, interested, proud, lonely, vulnerable, etc.
        - Level 3 (Tertiary): aroused, cheeky, free, joyful, curious, isolated, abandoned, etc.

        Work through the following steps:
        1. Review the previous question and answer pairs to understand the context.
        2. Analyze the detected moods and their confidence levels to identify the current depth level.
        3. If the current mood is at a shallow depth (level 1 or 2), formulate a question that will help identify a MORE SPECIFIC emotion at the next depth level.
        4. Use the wheel of emotions structure to guide your questioning toward deeper, more nuanced emotions.
        5. Ensure the question is open-ended and encourages the user to share more details about their emotional state.
        6. DO NOT ask direct questions like "Are you feeling X?" - instead, ask about situations, thoughts, or physical sensations that reveal deeper emotions.
        
        STRATEGY FOR GOING DEEPER:
        - If at primary level (e.g., "happy"): Ask about the QUALITY or FLAVOR of that happiness (peaceful? excited? proud?)
        - If at secondary level (e.g., "content"): Ask about specific ASPECTS or NUANCES (what makes it feel free vs joyful?)
        - Focus on concrete examples, recent moments, or physical sensations to elicit more specific emotional language
    
        USER QUESTION AND ANSWER HISTORY:
        {qa_history}

        DETECTED MOODS AND CONFIDENCE LEVELS:
        {mood_history}

        WHEEL OF EMOTIONS (for reference):
        {wheel_of_emotions}

        RESPONSE FORMAT:
        "<next question to ask the user to drill deeper into their emotional state>"
    """
    prompt_filled = prompt.format(
        current_depth=current_depth,
        current_depth_name=current_depth_name,
        target_depth=min(current_depth + 1, max_depth),
        target_depth_name=target_depth_name,
        qa_history="\n".join([f"Q: {q}\nA: {a}" for q, a in qa_pairs]),
        mood_history="\n".join(
            [f"Mood: {mood}, Confidence: {confidence}" for mood, confidence in moods]
        ),
        wheel_of_emotions=get_wheel_of_emotions(),
    )
    print(prompt_filled)

    try:
        response = get_gemini_client().models.generate_content(
            model="gemini-3-pro-preview",
            contents=[prompt_filled],
            config={
                "response_mime_type": "application/json",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Question generation failed: {e}")

    try:
        next_question = response.text.strip().strip('"')
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Question generation parsing failed: {e}"
        )

    return next_question
