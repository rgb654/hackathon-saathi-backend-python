from pydantic import BaseModel
from typing import List

class UpdateSkillsRequest(BaseModel):
    pidx: str              
    skills: List[str] 
