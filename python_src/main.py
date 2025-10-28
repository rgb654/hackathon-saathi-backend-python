from fastapi import FastAPI
from utils.embedder import get_skill_embedding
from models.participants import UpdateSkillsRequest
from db.participants import insert_participant
from db.hackathons import insert_hackathon
from utils.hackathon_context import generate_hackathon_skills
from utils.recommender import recommend_teammates

app = FastAPI()

@app.get("/health")
def healthcheck():
    return {"status": "ok"}

@app.get("/updateSkills")
def update_skills_list(payload: UpdateSkillsRequest):
    embedding = get_skill_embedding(payload.skills)
    return(embedding)
    # insert_participant(payload.idx, embedding)

@app.get("/updateHackathonSkills")
def update_hackathon_skills(idx:str):
    target_hackathon_skills = generate_hackathon_skills(idx).get("target_skills")
    embedding = get_skill_embedding(target_hackathon_skills)
    return(target_hackathon_skills)
    # insert_hackathon(idx,embedding)

@app.get("/getRecommendations")
def get_recommendations(hidx:str,uidx:str, top_k: int = 10):
    top_k_recommendations = recommend_teammates(hidx,uidx,top_k)
    return top_k_recommendations