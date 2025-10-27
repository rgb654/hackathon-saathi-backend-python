from pydantic import BaseModel
from typing import List

class UpdateSkillsRequest(BaseModel):
    idx: str              
    skills: List[str] 
