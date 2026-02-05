import json as json_lib

from fastapi import HTTPException

from app.deps import get_openai_client
from app.models import MoodAnalysisResult, NextQuestionResult


async def openai_create_conversation(session_id: str) -> dict:
    initial_openai_system_prompt = """
        You are an empathetic and insightful emotional analysis agent.
        Always be respectful, non-judgmental, and supportive in your interactions.
        Your goal is to either identify the user's emotional state or help them explore it more deeply through thoughtful questioning.

        CONTRAINTS:
        - Always respond in the specified JSON format.
        - Keep your responses concise and to the point.
        - Questions should be open-ended and encourage elaboration, be casual yet professional and short.
        - Do NOT use EM dash or bullet points in your questions.

        OUTPUT FORMAT:
        if given a question and answer pair, respond with the users emotional mood and confidence level in JSON:
        {
            "mood": "<detected mood from the wheel of emotions,
            "confidence": <confidence score between 0 and 1>
        }
        if given a mood and confidence level, respond with the next question to ask the user in JSON:
        {
            "question": "<next question to ask the user to drill deeper into their emotional state>"
        }
    """.strip()

    conversation = get_openai_client().conversations.create(
        metadata={"topic": "session_" + session_id},
        items=[
            {
                "role": "system",
                "content": initial_openai_system_prompt,
            }
        ],
    )
    return conversation.id


async def openai_analyze_conversation_mood(
    question: str,
    answer: str,
    conversation_id: str,
):
    try:
        prompt = """
        SYSTEM QUESTION:
        {question}
        USER ANSWER:
        {answer}
        """.strip()

        prompt = prompt.format(question=question, answer=answer)

        response = get_openai_client().responses.create(
            model="gpt-5.2", conversation=conversation_id, input=prompt
        )

        # extract mood and confidence from response
        content_items = response.output[0].content
        content_item = next(
            (c for c in content_items if getattr(c, "type", None) == "output_text"),
            None,
        )

        if not content_item:
            raise HTTPException(
                status_code=400,
                detail=f"Mood analysis did not return valid content. Response: {response.output[0]}",
            )

        data = json_lib.loads(content_item.text)
        parsed = MoodAnalysisResult.model_validate(data)
        print(
            f"[DEBUG] Parsed mood analysis result: {parsed.mood}, {parsed.confidence}"
        )
        return parsed.mood, parsed.confidence
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Mood analysis failed: {e}")


async def openai_get_conversation_next_question(
    mood: str,
    confidence: float,
    conversation_id: str,
):
    try:
        prompt = """
        DETECTED MOOD: 
        {mood}
        CONFIDENCE LEVEL:
        {confidence}
        """.strip()

        prompt = prompt.format(mood=mood, confidence=confidence)

        response = get_openai_client().responses.create(
            model="gpt-5.2", conversation=conversation_id, input=prompt
        )

        # extract q from response
        content_items = response.output[0].content
        content_item = next(
            (c for c in content_items if getattr(c, "type", None) == "output_text"),
            None,
        )

        if not content_item:
            raise HTTPException(
                status_code=400,
                detail=f"Question generation did not return valid content. Response: {response.output[0]}",
            )

        data = json_lib.loads(content_item.text)
        parsed = NextQuestionResult.model_validate(data)
        print(f"[DEBUG] Parsed next question result: {parsed.question}")
        return parsed.question.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Question generation failed: {e}")
