from pymongo import MongoClient
from bson import ObjectId

# Global connection state
_mongo_connected = False
_client = None
_db = None

def connect_mongo():
    global _mongo_connected, _client, _db
    
    import os
    from dotenv import load_dotenv
    
    load_dotenv()

    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DB = os.getenv("MONGO_DB")

    
    _client = MongoClient(MONGO_URI)
    _db = _client[MONGO_DB]
    _mongo_connected = True
    print("✅ Mongo connected successfully")
    

def ensure_mongo_connected():
    """Ensure MongoDB connection is established"""
    global _mongo_connected
    if not _mongo_connected:
        connect_mongo()



def get_mongo_db():
    """Get MongoDB database instance"""
    ensure_mongo_connected()
    return _db

def get_hackathons_collection():
    """Get hackathons collection"""
    ensure_mongo_connected()
    return _db["hackathons"]

def get_users_collection():
    """Get users collection"""
    ensure_mongo_connected()
    return _db["users"]