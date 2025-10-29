from pymilvus import connections
from pymilvus import Collection

_milvus_connected = False

def connect_milvus():
    global _milvus_connected
    
    import os
    from dotenv import load_dotenv
    load_dotenv()

    MILVUS_URI = os.getenv("MILVUS_URI")
    MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
    
    connections.connect(
        alias="default",
        uri=MILVUS_URI,
        token=MILVUS_TOKEN
    )
    _milvus_connected = True
    print("✅ Milvus connected successfully")

def ensure_milvus_connected():
    global _milvus_connected
    if not _milvus_connected:
        connect_milvus()
    
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

def update_hackathon(hid: str, new_embedding: list):
    ensure_milvus_connected()
    collection = get_hackathon_collection()
    collection.load()
    
    # Check if hid exists
    existing = collection.query(
        expr=f'hid == "{hid}"',
        output_fields=["hid"]
    )
    
    if not existing:
        raise ValueError(f"Hackathon with hid '{hid}' not found")
    
    # Update the embedding
    collection.delete(expr=f'hid == "{hid}"')
    collection.insert([[hid], [new_embedding]])
    collection.flush()

def update_participant(pid: str, new_embedding: list):
    ensure_milvus_connected()
    collection = get_participant_collection()
    collection.load()
    
    # Check if pid exists
    existing = collection.query(
        expr=f'pid == "{pid}"',
        output_fields=["pid"]
    )
    
    if not existing:
        raise ValueError(f"User with pid '{pid}' not found")
    
    # Update the embedding
    collection.delete(expr=f'pid == "{pid}"')
    collection.insert([[pid], [new_embedding]])
    collection.flush()

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