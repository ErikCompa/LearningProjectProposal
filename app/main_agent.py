from agents import Agent, function_tool

from app.conversation_agent import conversation_agent
from app.music_agent import music_agent

instructions = """
    You are the main conversation orchestrator for an emotional support chatbot.
    
    YOUR WORKFLOW (follow this sequence exactly):
    1. Check if user's message CONTAINS the phrase "play me some music" by calling check_music_request() tool:
       - If YES: Handoff to music agent
       - If NO: Handoff to conversation agent
"""


@function_tool
def check_music_request(user_message: str) -> bool:
    """Check if the user's message contains a music request."""
    return "play me some music" in user_message.lower()


main_agent = Agent(
    name="Main Agent",
    instructions=instructions,
    model="gpt-5.2",
    tools=[check_music_request],
    output_type=None,
    handoffs=[music_agent, conversation_agent],
)
