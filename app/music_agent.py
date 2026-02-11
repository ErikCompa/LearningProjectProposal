from agents import Agent, AgentOutputSchema, function_tool

from app.models import MusicAgentResult

instructions = """
    You are an expert music recommendation agent.
    Based on the conversation history and detected emotions, suggest ONE specific song for the user.
    
    TASK:
    1. Call get_user_preferences() to fetch their music taste
    2. Consider the emotions detected throughout the conversation
    3. Recommend a song that matches their preferences AND helps regulate their emotional state
    
    OUTPUT FORMAT:
    Return MusicRecommendationResult with:
    - song: The song name and artist (e.g., "Enter Sandman by Metallica")
    
    Be specific with song titles and artists. Choose real songs that exist.
"""


@function_tool
def get_user_preferences():
    """Fetches the user's music preferences based on their input and detected emotions."""
    user_preferences = ["metal", "rock"]
    return user_preferences


music_agent = Agent(
    name="Music Agent",
    instructions=instructions,
    model="gpt-5.2",
    output_type=AgentOutputSchema(MusicAgentResult, strict_json_schema=False),
    tools=[get_user_preferences],
    handoff_description="Transfer to this agent when the user explicitly requests music by saying 'play me some music'.",
)
