from pymilvus import Collection

def get_hackathon_collection():
    return Collection("hackathons")

def insert_hackathon(hid: str, embedding: list):
    collection = get_hackathon_collection()
    collection.insert([[hid], [embedding]])
    collection.flush()  # ensure data is persisted