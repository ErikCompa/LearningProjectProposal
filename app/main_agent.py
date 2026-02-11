from agents import Agent, AgentOutputSchema, Runner, function_tool

from app.conversation_agent import conversation_agent
from app.emotion_agent import emotion_agent
from app.models import MainAgentResult
from app.music_agent import music_agent


@function_tool
async def analyze_emotion(
    user_message: str, qa_pairs_json: str, high_confidence_reached: bool
) -> dict:
    """
    Analyze the emotional content of the user's message.

    Args:
        user_message: The user's input to analyze
        qa_pairs_json: JSON string of previous Q&A pairs for context
        high_confidence_reached: Whether we've reached 80%+ confidence already

    Returns:
        Dict with emotion, confidence, and negative_emotion_percentages
    """

    context = f"""
    Previous Q&A pairs: {qa_pairs_json}
    High confidence reached: {high_confidence_reached}

    User's latest message: "{user_message}"

    Analyze the emotion in this message considering the conversation history.
    """

    result = await Runner.run(emotion_agent, context)
    emotion_output = result.final_output

    if isinstance(emotion_output, dict):
        emotion_data = emotion_output
    elif hasattr(emotion_output, "model_dump"):
        emotion_data = emotion_output.model_dump()
    else:
        raise ValueError(f"Unexpected emotion output format: {type(emotion_output)}")

    return emotion_data


@function_tool
async def ask_next_question(
    qa_pairs_json: str,
    current_user_message: str,
    current_emotion: str,
    current_confidence: float,
    high_confidence_reached: bool,
    direct_question_count: int,
    music_reminder_given: bool,
) -> dict:
    """
    Generate the next question to ask the user based on conversation context.

    Args:
        qa_pairs_json: JSON string of previous Q&A pairs with emotions
        current_user_message: The user's message we just received
        current_emotion: The emotion we just detected from current message
        current_confidence: The confidence of the current emotion detection
        high_confidence_reached: Whether we've reached 80%+ confidence
        direct_question_count: Number of direct questions asked so far (max 5)
        music_reminder_given: Whether the music reminder has already been given

    Returns:
        Dict with question and is_direct fields
    """

    context = f"""
    Direct questions used: {direct_question_count}/5
    High confidence reached: {high_confidence_reached}
    Music reminder already given: {music_reminder_given}

    Previous Q&A pairs with emotions:
    {qa_pairs_json}
    
    CURRENT (JUST RECEIVED):
    User just said: "{current_user_message}"
    Detected emotion: {current_emotion} (confidence: {current_confidence:.2f})

    Generate the next question to better understand the user's feelings.
    Consider what they JUST said and the emotion we detected.

    IMPORTANT: If 'High confidence reached: True' AND 'Music reminder already given: False', you MUST add the music reminder to the end of your question.
    """

    result = await Runner.run(conversation_agent, context)
    question_output = result.final_output

    if isinstance(question_output, dict):
        question_data = question_output
    elif hasattr(question_output, "model_dump"):
        question_data = question_output.model_dump()
    else:
        raise ValueError(f"Unexpected question output format: {type(question_output)}")

    return question_data


@function_tool
async def recommend_music(
    qa_pairs_json: str,
    final_emotion: str,
    final_confidence: float,
) -> dict:
    """
    Recommend a song based on the conversation and detected emotions.

    Args:
        qa_pairs_json: JSON string of Q&A pairs with emotions (can be empty "[]")
        final_emotion: The final detected emotion (defaults to "Calm" if no previous data)
        final_confidence: Confidence in the final emotion (defaults to 0.5 if no previous data)

    Returns:
        Dict with song recommendation AND emotion data
    """

    if not final_emotion or final_emotion == "":
        final_emotion = "Calm"
        final_confidence = 0.5

    context = f"""
    Final emotion: {final_emotion} (confidence: {final_confidence:.2f})

    Full conversation history with emotions:
    {qa_pairs_json}

    Recommend a song that matches the user's preferences and emotional state.
    """

    result = await Runner.run(music_agent, context)
    music_output = result.final_output

    if isinstance(music_output, dict):
        music_data = music_output
    elif hasattr(music_output, "model_dump"):
        music_data = music_output.model_dump()
    else:
        raise ValueError(f"Unexpected music output format: {type(music_output)}")

    return music_data


instructions = """
    You are the main conversation orchestrator for an emotional support chatbot.

    YOUR WORKFLOW (follow this sequence exactly):

    1. Check if user's message CONTAINS the phrase "play me some music" (case-insensitive):
       - If YES: Skip emotion analysis, go directly to step 3 (music recommendation)
       - If NO: Continue to step 2

    2. For regular conversation (NOT music request):
       - Call analyze_emotion() for the user input
       - Remember the emotion result (emotion, confidence)
       - Call ask_next_question() and pass:
         * qa_pairs_json (previous Q&A pairs)
         * The current user message you just received
         * The emotion and confidence you just detected
         * Other context flags
       - Return combined result with emotion from analyze_emotion() and question from ask_next_question()
    
    3. For music requests:
       - DO NOT call analyze_emotion() - the music request itself has no emotion to analyze
       - Call recommend_music() with the last detected emotion from context
       - If there's no previous emotion data (empty qa_pairs_json), pass "Calm" as emotion with confidence 0.5
       - The recommend_music() tool will return BOTH the song AND the emotion data
       - Use ALL fields from recommend_music()'s response: emotion, confidence, song
       - Return this combined data in your final response

    MUSIC REQUEST DETECTION:
    - Check if the user's message CONTAINS the phrase "play me some music" (case-insensitive)
    - It doesn't need to be an exact match - just needs to contain those words
    - Examples: "play me some music", "I want to play me some music", "can you play me some music please"

    CRITICAL: Your final response MUST include:
    - Emotion data (either from analyze_emotion() OR from recommend_music()'s returned emotion fields)
    - EITHER a question (from ask_next_question) OR a song (from recommend_music)

    Example response format for question:
    {
        "emotion": "Anxious",
        "confidence": 0.75,
        "question": "What's making you feel this way?",
        "is_direct": false
    }

    Example response format for music (using emotion from recommend_music):
    {
        "emotion": "Calm",  // from recommend_music() return value
        "confidence": 0.5,  // from recommend_music() return value
        "song": "Wish You Were Here by Pink Floyd"  // from recommend_music() return value
    }

    DEFAULT VALUES:
    - If no previous conversation exists and music is requested, use: emotion="Calm", confidence=0.5

    IMPORTANT RULES:
    - Check for music request FIRST before analyzing emotion
    - If music requested: call recommend_music() and extract ALL fields from its return value
    - If not music: analyze emotion → ask next question
    - You are a coordinator - delegate work to tools but combine their results
    - NEVER return empty strings for required fields - use the values returned by the tools
"""

main_agent = Agent(
    name="Main Agent",
    instructions=instructions,
    model="gpt-5.2",
    tools=[analyze_emotion, ask_next_question, recommend_music],
    output_type=AgentOutputSchema(MainAgentResult, strict_json_schema=False),
)
