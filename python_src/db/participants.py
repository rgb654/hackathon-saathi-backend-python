from pymilvus import Collection

def get_participant_collection():
    return Collection("participants")

def insert_participant(pid: str, embedding: list):
    collection = get_participant_collection()
    collection.insert([[pid], [embedding]])
    collection.flush()  # ensure data is persisted
