import heapq
import numpy as np
from pymilvus import Collection
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity

def _get_collection(name: str) -> Collection:
    return Collection(name)

def _get_embedding(collection: Collection, key_field: str, key_val: str, emb_field: str = "embedding") -> np.ndarray:
    """
    Fetch a single embedding vector from a collection given its ID.
    """
    results = collection.query(
        expr=f'{key_field} == "{key_val}"',
        output_fields=[emb_field]
    )
    if not results:
        raise ValueError(f"No entry found in {collection.name} with {key_field}={key_val}")
    return np.array(results[0][emb_field], dtype=float)

# def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
#     """
#     Compute cosine similarity between two vectors.
#     """
#     if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
#         return 0.0
#     return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))

def recommend_teammates(hidx: str, uidx: str, top_k: int = 10) -> List[Tuple[float, str]]:
    """
    Recommend teammates for a participant in a hackathon.

    Args:
        hidx: Hackathon ID (hid in hackathons collection).
        uidx: User/participant ID (pid in participants collection).
        top_k: Number of top teammates to return.

    Returns:
        List of tuples (similarity, uid) sorted by highest similarity first.
    """
    participants = _get_collection("participants")
    hackathons = _get_collection("hackathons")

    # Get target embedding (hackathon) and current participant embedding
    target_emb = _get_embedding(hackathons, "hid", hidx)
    user_emb_a = _get_embedding(participants, "pid", uidx)

    # Priority queue (min-heap), we’ll invert similarity for max behavior
    heap = []

    # Load all embeddings from participants
    results = participants.query(
        expr=f'pid != "{uidx}"',
        output_fields=["pid", "embedding"]
    )

    for row in results:
        uid_b = row["pid"]
        user_emb_b = np.array(row["embedding"], dtype=float)

        # Team vector = normalized(user_a + user_b)
        team_vec = user_emb_a + user_emb_b
        if np.linalg.norm(team_vec) > 0:
            team_vec = team_vec / np.linalg.norm(team_vec)

            sim = cosine_similarity(team_vec.reshape(1, -1), target_emb.reshape(1, -1))[0][0]

        # Push into heap as negative similarity (to simulate max-heap)
        heapq.heappush(heap, (-sim, uid_b))

    # Extract top_k results
    top_results = []
    for _ in range(min(top_k, len(heap))):
        sim, uid_b = heapq.heappop(heap)
        top_results.append((-sim, uid_b))  # restore positive similarity

    return top_results
