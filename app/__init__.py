from app.conversation_agent import conversation_agent
from app.emotion_agent import emotion_agent
from app.music_agent import music_agent
from app.orchestration_agent import orchestration_agent

# config handoffs - orchestration routes to conversation/music, not emotion
orchestration_agent.handoffs = [music_agent, conversation_agent]
conversation_agent.handoffs = [music_agent]

__all__ = ["emotion_agent", "conversation_agent", "music_agent", "orchestration_agent"]
