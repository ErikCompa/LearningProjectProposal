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

        Work through the following steps:
        1. Review the previous question and answer pairs to understand the context.
        2. Analyze the latest answer in relation to the latest question.
        3. Map the emotional cues from the previous and current answers to the wheel of emotions.
        4. Determine the most fitting mood from the wheel of emotions, the deeper you can go into the layers the better, along with a confidence score.
        5. Provide the mood and confidence score in the specified response format.
        
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
            "mood": "<detected mood from the wheel of emotions>",
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
    qa_pairs: list[tuple[str, str]], moods: list[tuple[str, float]]
) -> str:
    next_question = ""

    prompt = """
        You are an expert in emotional analysis and questioning techniques using the wheel of emotions framework.

        Based on the previous question and answer pairs, as well as the detected moods and their confidence levels,
        generate the next most effective question to better understand the user's emotional state.

        Work through the following steps:
        1. Review the previous question and answer pairs to understand the context.
        2. Analyze the detected moods and their confidence levels to identify gaps in understanding.
        3. Formulate a question that targets those gaps, while not being direct or leading, but aiming to elicit responses that will clarify the user's emotional state.
        4. Ensure the question is open-ended and encourages the user to share more about their feelings.
        5. Provide the next question in the specified response format.
    
        USER QUESTION AND ANSWER HISTORY:
        {qa_history}

        DETECTED MOODS AND CONFIDENCE LEVELS:
        {mood_history}

        WHEEL OF EMOTIONS:
        {wheel_of_emotions}

        RESPONSE FORMAT:
        "<next question to ask the user to better understand their emotional state>"
    """
    prompt_filled = prompt.format(
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
