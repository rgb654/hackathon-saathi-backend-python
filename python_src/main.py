import os
from dotenv import load_dotenv

load_dotenv("python_src/.env")
load_dotenv()  # Also try loading from root directory

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .utils.embedder import get_skill_embedding
from .models.participants import UpdateSkillsRequest
from .db.milvus_client import insert_participant, insert_hackathon
from .utils.hackathon_context import generate_hackathon_skills
from .utils.recommender import recommend_teammates

app = FastAPI()

# Add CORS middleware to allow requests from Node.js backend and frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js frontend
        "http://localhost:8000",  # Node.js backend
        "http://localhost:5000",  # Alternative ports
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:5000",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.get("/health")
def healthcheck():
    return {"status": "ok"}

@app.post("/updateSkills")
def update_skills_list(payload: UpdateSkillsRequest):
    try:
        embedding = get_skill_embedding(payload.skills)
        insert_participant(pid=str(payload.pidx), embedding=embedding)
        return {"status": "success", "message": f"Skills updated for user {payload.pidx}"}
    except Exception as e:
        print(f"❌ Error updating skills for {payload.pidx}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to update skills: {str(e)}")

@app.post("/updateHackathonSkills")
def update_hackathon_skills(payload: dict):
    """
    Sync a hackathon to Milvus vector database.
    Expected payload: {"hackathonId": "<mongo_id>", "skills": ["skill1", "skill2", ...] (optional)}
    If skills not provided, will use AI to generate them from hackathon context.
    """
    try:
        hidx = payload.get("hackathonId")
        provided_skills = payload.get("skills")
        
        if not hidx:
            raise HTTPException(status_code=400, detail="hackathonId is required")
        
        # If skills provided, use them directly; otherwise generate using AI
        if provided_skills and len(provided_skills) > 0:
            target_hackathon_skills = provided_skills
            print(f"Using provided skills for hackathon {hidx}: {target_hackathon_skills}")
        else:
            print(f"Generating AI skills for hackathon {hidx}...")
            out = generate_hackathon_skills(hidx)
            target_hackathon_skills = out.get("target_skills")
            print(f"Generated skills: {target_hackathon_skills}")
        
        # Create embedding and insert into Milvus
        embedding = get_skill_embedding(target_hackathon_skills)
        insert_hackathon(hid=str(hidx), embedding=embedding)
        
        return {
            "status": "success",
            "hackathonId": hidx,
            "skills": target_hackathon_skills,
            "message": f"Hackathon {hidx} synced successfully to AI database"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error syncing hackathon {payload.get('hackathonId')}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to sync hackathon: {str(e)}")

@app.get("/getRecommendations")
def get_recommendations(hidx:str,uidx:str, top_k: int = 10):
    top_k_recommendations = recommend_teammates(hidx=hidx,pidx=uidx,top_k=top_k)
    return top_k_recommendations

@app.get("/getTeammateRecommendations")
def get_teammate_recommendations(uidx: str, top_k: int = 50):
    """
    Pure AI-based teammate recommendations without hackathon context.
    Finds users with complementary skills using vector similarity.
    """
    try:
        from pymilvus import Collection
        from .db.milvus_client import get_participant_collection, ensure_milvus_connected
        
        ensure_milvus_connected()
        participants = get_participant_collection()
        participants.load()
        
        # Get user's skill embedding
        participant_data = participants.query(
            expr=f'pid == "{uidx}"',
            output_fields=["embedding"]
        )
        
        if not participant_data:
            raise HTTPException(status_code=404, detail=f"User {uidx} not found in AI database. Please sync their skills first.")
        
        user_embedding = participant_data[0]["embedding"]
        
        # Search for similar users (complementary skills)
        # Using COSINE similarity to find users with related but different skill sets
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        results = participants.search(
            data=[user_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k + 1,  # Get extra to exclude self
            expr=f'pid != "{uidx}"',  # Exclude current user
            output_fields=["pid"]
        )
        
        # Process results
        recommendations = []
        for hits in results:
            for hit in hits:
                if hit.entity.get("pid") != uidx:
                    recommendations.append({
                        "pid": hit.entity.get("pid"),
                        "ai_score": float(hit.score),  # Similarity score
                        "distance": float(hit.distance)
                    })
        
        return recommendations[:top_k]
        
    except Exception as e:
        print(f"❌ Error getting AI recommendations: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get AI recommendations: {str(e)}")