# import heapq
import numpy as np
# from pymilvus import Collection
# from typing import List, Tuple
# from sklearn.metrics.pairwise import cosine_similarity
from ..db.milvus_client import get_hackathon_collection, get_participant_collection



def recommend_teammates(pidx: str, hidx: str, top_k: int = 10):
    participants = get_participant_collection()
    hackathons = get_hackathon_collection()
    
    participants.load()
    hackathons.load()
    
    # Get participant embedding (user_emb_a)
    participant_data = participants.query(
        expr=f'pid == "{pidx}"',
        output_fields=["embedding"]
    )
    
    # Get hackathon embedding (target_emb)
    hackathon_data = hackathons.query(
        expr=f'hid == "{hidx}"',
        output_fields=["embedding"]
    )
    
    if not participant_data:
        raise ValueError(f"Participant {pidx} not found")
    if not hackathon_data:
        raise ValueError(f"Hackathon {hidx} not found")
    
    user_emb_a = participant_data[0]["embedding"]
    target_emb = hackathon_data[0]["embedding"]
    
    # Step 2: Calculate adjusted target embedding
    user_emb_a_np = np.array(user_emb_a)
    target_emb_np = np.array(target_emb)
    
    # print(np.linalg.norm(user_emb_a_np), np.linalg.norm(target_emb_np))
    # Perform vector subtraction: target_emb - user_emb_a
    user_emb_b_np = target_emb_np - user_emb_a_np

    # Normalize required user_emb_b vector
    norm = np.linalg.norm(user_emb_b_np)
    if norm == 0:
        normalized_user_emb_b_np = user_emb_b_np
    else:
        normalized_user_emb_b_np = user_emb_b_np / norm

    # print(np.linalg.norm(normalized_user_emb_b_np))

    # Convert back to list for Milvus search
    user_emb_b = normalized_user_emb_b_np.tolist()
    
    # Step 3: Search for similar participants (excluding the original pidx)
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10}  # Adjust based on your index
    }
    
    # Search in participants collection (excluding the original participant)
    results = participants.search(
        data=[user_emb_b],  # Search with adjusted target
        anns_field="embedding",
        param=search_params,
        limit=top_k + 1,  # Get extra to account for potential self-match
        expr=f'pid != "{pidx}"',  # Exclude the original participant
        output_fields=["pid"]  # Return participant IDs
    )
    
    # Process results (filter out any accidental self-matches)
    similar_participants = []
    for hits in results:
        for hit in hits:
            if hit.entity.get("pid") != pidx:  # Double-check exclusion
                similar_participants.append({
                    "pid": hit.entity.get("pid"),
                    "similarity_score": hit.score,
                    "distance": hit.distance
                })
    
    # Return top_k results
    return similar_participants[:top_k]