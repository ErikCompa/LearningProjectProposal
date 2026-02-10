from agents import Agent, function_tool


@function_tool
def check_music_request(user_message: str) -> bool:
    """
    Check if the user's message contains an exact request to play music.

    Args:
        user_message: The user's input message to check

    Returns:
        True if music was requested (user said 'play me some music'), False otherwise
    """
    return "play me some music" in user_message.lower()


instructions = """
    You are an orchestration agent that routes conversations to the appropriate specialist agent.
    
    YOUR TASK:
    1. Call check_music_request tool with the user's message
    2. Based on the result, decide which agent to hand off to:
       - If check_music_request returns True: Hand off to Music Agent
       - If check_music_request returns False: Hand off to Conversation Agent
    
    You are a router only. Always hand off after checking - never generate responses yourself.
    
    Note: Emotion analysis happens before you, so just focus on routing based on music request.
"""

orchestration_agent = Agent(
    name="Orchestration Agent",
    instructions=instructions,
    model="gpt-5.2",
    tools=[check_music_request],
    handoffs=[],
)
