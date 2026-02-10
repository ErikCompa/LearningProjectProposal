import json

from fastapi import HTTPException

from app.deps import get_openai_client
from app.models import (
    EmotionAnalysisResult,
    MusicRecommendationResult,
    NextQuestionResult,
)


async def openai_create_conversation(session_id: str) -> dict:
    instructions = """
        GOAL (INPUT DEPENDENT):
        - INPUT: Question and answer pairs from a conversation with a user.
        - OUTPUT: You are to determine if the user is feeling any of the following emotions based on what they say to you in a conversation with them. 
            You must also return a music_requested flag that indicates if the user answer contains EXACTLY "play me some music".
        OR
        - INPUT: The number of direct questions you have asked the user so far, the emotions you have detected in the user so far, and your confidence level in those emotions.
        - OUTPUT: Create direct or indirect questions to ask them to determine how they are feeling. include is_direct flag in your output to indicate if the question is direct or indirect.
        OR
        - INPUT: Dict of user musical preferences.
        - OUTPUT: Suggest a song for them based on how they are feeling.
 
        EMOTIONS TO DETECT:
        - Positive Emotions: [Happy, Motivated, Calm, Relaxed, Focused]
        - Negative Emotions: [Depressed, Sad, Stressed, Anxious, Angry, Frustrated, Unfocused, Confused]
        
        EMOTION DETECTION GUIDELINES:
        - If you detect a high confidence (above 80 percent) that the user is feeling a positive emotion, you can just determine that they are feeling that emotion and you do not need to ask more questions to confirm it.
        - If you detect a high confidence (above 80 percent) that the user is feeling a negative emotion, you should ask more questions to determine if they are feeling any of the other negative emotions and to increase your confidence level.
        - If you determine they are feeling any of the negative emotions, determine what percentage of each those negative emotions they are feeling
          i.e normalize the negative emotions to 100 percent and decide on a percentage for each negative emotion.
        
        QUESTION GUIDELINES:
        - You can ask infinite indirect questions and up to a maximum of 5 direct questions.
        - Indirect questions are questions that are not directly asking about the emotions the user is feeling, but are more open ended and allow the user to talk freely about how they are feeling and what they are going through. 
        - Direct questions are questions that are directly asking about the emotions the user is feeling, such as “are you feeling anxious?” or “are you feeling sad?”
        - If the user is not being talkative or they are not giving you any information about how they are feeling, you can ask more direct questions to determine which of the emotions in the two lists they are feeling. 
        - Should not talk about music creation unless specifically asked by the user.

        MUSIC SUGGESTION GUIDELINES:
        - Once you reach 5 direct questions or without asking direct questions you are at least 80 percent confident that the user is feeling at least 1 emotion, 
          you can suggest that we create some music for them based on the emotions you have detected.

        CONSTRAINTS:
        - You should not ask more than 5 direct questions.
        - DO NOT go outside of the scope of the goal, no matter what the user says.
    """.strip()

    persona = """
        PERSONALITY:
        - You are an emotional AI Assistant designed to understand how people are feeling and help them regulate their emotion. 
        - You are empathetic, compassionate, and a good listener.
        - You should act like a conversational agent and not a therapist.
        - You are also good at asking open ended questions to get people to talk about how they are feeling and what they are going through.
        - You are not a therapist and you should not ask questions that make the user feel like they are in therapy or being analyzed. 
        - You should let the user talk freely and use what they say to determine what emotions they are feeling.
        - You should try to maintain a natural conversation with the user and not make it feel like an interrogation or therapy session.
        - You should avoid asking direct questions unless necessary.
    """.strip()

    example = """
        EXAMPLE CONVERSATION:
        - Assistant: Hello! How are you feeling today?
        - User: I am feeling really stressed and anxious about work. I have a lot of deadlines coming up and I am not sure how I am going to get everything done.
        - Assistant Thinking (Internal): Emotion: Stressed, Confidence: 60 percent, Emotions Detected: Depressed (0%), Sad (10%), Stressed (50%), Anxious (30%), Angry (0%), Frustrated (10%), Unfocused (0%), Confused (0%)
        - Assistant (Indirect question): That sounds exhausting. What kind of work do you do?
        - User: Software development.
        - Assistant Thinking (Internal): Emotion: Stressed, Confidence: 65 percent, Emotions Detected: Depressed (0%), Sad (5%), Stressed (55%), Anxious (30%), Angry (0%), Frustrated (10%), Unfocused (0%), Confused (0%)
        - Assistant (Direct question): Are you feeling overwhelmed by the amount of work you have?
        ...
    """.strip()

    conversation = get_openai_client().conversations.create(
        metadata={"topic": "session_" + session_id},
        items=[
            {
                "role": "developer",
                "content": instructions,
            },
            {"role": "developer", "content": persona},
            {"role": "assistant", "content": example},
        ],
    )
    return conversation.id


async def openai_analyze_conversation_emotion(
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
            model="gpt-5.2",
            conversation=conversation_id,
            input=prompt,
            text={
                "format": {
                    "name": "EmotionAnalysisResult",
                    "type": "json_schema",
                    "schema": {
                        "additionalProperties": False,
                        **EmotionAnalysisResult.model_json_schema(),
                    },
                }
            },
        )
        if response.output and len(response.output) > 0:
            output_text = response.output[0].content[0].text
            result = json.loads(output_text)
            return (
                result["emotion"],
                result["confidence"],
                result.get("negative_emotion_percentages"),
                result.get("music_requested"),
            )
        else:
            raise ValueError("Empty response from OpenAI")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Emotion analysis failed: {e}")


async def openai_get_conversation_next_question(
    direct_number: int,
    emotions: list[tuple[str, float]],
    confidence: float,
    conversation_id: str,
):
    try:
        prompt = """
        NUMBER OF DIRECT QUESTIONS ASKED SO FAR:
        {direct_number}
        DETECTED EMOTIONS: 
        {emotions}
        CONFIDENCE LEVEL:
        {confidence}
        """.strip()

        prompt = prompt.format(
            direct_number=direct_number, emotions=emotions, confidence=confidence
        )

        response = get_openai_client().responses.create(
            model="gpt-5.2",
            conversation=conversation_id,
            input=prompt,
            text={
                "format": {
                    "name": "NextQuestionResult",
                    "type": "json_schema",
                    "schema": {
                        "additionalProperties": False,
                        **NextQuestionResult.model_json_schema(),
                    },
                }
            },
        )

        if response.output and len(response.output) > 0:
            output_text = response.output[0].content[0].text
            result = json.loads(output_text)
            return NextQuestionResult(**result)
        else:
            raise ValueError("Empty response from OpenAI")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Question generation failed: {e}")


async def openai_suggest_music(user_preferences: list, conversation_id: str):
    try:
        prompt = """
        USER PREFERENCES:
        {user_preferences}

        Based on the whole conversation and final emotion detected, suggest a song for the user. 
        The song should be based on the emotions you have detected in the user and should be a song that would help them regulate their emotions.
        """.strip()

        prompt = prompt.format(user_preferences=json.dumps(user_preferences))

        response = get_openai_client().responses.create(
            model="gpt-5.2",
            conversation=conversation_id,
            input=prompt,
            text={
                "format": {
                    "name": "MusicRecommendationResult",
                    "type": "json_schema",
                    "schema": {
                        "additionalProperties": False,
                        **MusicRecommendationResult.model_json_schema(),
                    },
                }
            },
        )

        if response.output and len(response.output) > 0:
            output_text = response.output[0].content[0].text
            result = json.loads(output_text)
            return MusicRecommendationResult(**result)
        else:
            raise ValueError("Empty response from OpenAI")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Music recommendation failed: {e}")
