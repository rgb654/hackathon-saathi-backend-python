from pydantic import BaseModel
from typing import List

class HackathonDreamTeam(BaseModel):
    hackathon_name: str
    required_skills: List[str]