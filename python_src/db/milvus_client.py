from pymilvus import connections
from pymilvus import Collection

_milvus_connected = False

def connect_milvus():
    import os
    from dotenv import load_dotenv
    load_dotenv()

    connections.connect(
        alias="default",
        uri=os.getenv("MILVUS_URI"),
        token=os.getenv("MILVUS_TOKEN")
    )

def ensure_milvus_connected():
    global _milvus_connected
    if not _milvus_connected:
        connect_milvus()
        _milvus_connected = True
    
def get_hackathon_collection():
    ensure_milvus_connected()
    return Collection("hackathons")

def get_participant_collection():
    ensure_milvus_connected()
    return Collection("participants")

def insert_hackathon(hid: str, embedding: list):
    ensure_milvus_connected()
    collection = get_hackathon_collection()
    collection.insert([[hid], [embedding]])
    collection.flush()  # ensure data is persisted

def insert_participant(pid: str, embedding: list):
    ensure_milvus_connected()
    collection = get_participant_collection()

    collection.insert([[pid], [embedding]])
    collection.flush()  # ensure data is persisted
    
def batch_insert_hackathons(hids: list, embeddings: list):
    ensure_milvus_connected()
    collection = get_hackathon_collection()
    collection.insert([hids, embeddings])
    collection.flush()  # ensure data is persisted

def batch_insert_participants(pids: list, embeddings: list):
    ensure_milvus_connected()
    collection = get_participant_collection()
    collection.insert([pids, embeddings])
    collection.flush()  # ensure data is persisted

def delete_all_entries(collection_name: str):
    ensure_milvus_connected()
    collection = Collection(collection_name)
    collection.load()
    
    if collection_name == "participants":
        expr = "pid != ''"
    elif collection_name == "hackathons":
        expr = "hid != ''"
    else:
        expr = "1 == 1"  # Always true expression as fallback
    
    # Perform deletion
    delete_result = collection.delete(expr=expr)
    
    # Flush to ensure changes are persisted
    collection.flush()
    
    print(f"Deleted all entries from {collection_name}")
    return delete_result