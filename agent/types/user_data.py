from dataclasses import dataclass
from typing import Dict, Optional
from livekit.agents.voice import Agent

@dataclass
class UserData:
    """사용자 데이터를 저장하는 클래스"""
    agents: Dict[str, Agent]
    prev_agent: Optional[Agent]
    user_id: str 