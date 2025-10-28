from fastapi import FastAPI
from utils.embedder import get_skill_embedding
from models.participants import UpdateSkillsRequest
from db.milvus_client import insert_participant, insert_hackathon
from utils.hackathon_context import generate_hackathon_skills
from utils.recommender import recommend_teammates

app = FastAPI()

@app.get("/health")
def healthcheck():
    return {"status": "ok"}

@app.get("/updateSkills")
def update_skills_list(payload: UpdateSkillsRequest):
    embedding = get_skill_embedding(payload.skills)
    insert_participant(pid=str(payload.pidx), embedding=embedding)

@app.get("/updateHackathonSkills")
def update_hackathon_skills(hidx:str):
    out = generate_hackathon_skills(hidx)
    target_hackathon_skills = out.get("target_skills")
    embedding = get_skill_embedding(target_hackathon_skills)

    insert_hackathon(hid=str(hidx), embedding=embedding)

    return(target_hackathon_skills)

@app.get("/getRecommendations")
def get_recommendations(hidx:str,uidx:str, top_k: int = 10):
    top_k_recommendations = recommend_teammates(hidx,uidx,top_k)
    return top_k_recommendations