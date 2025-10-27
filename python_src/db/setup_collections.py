from pymilvus import FieldSchema, CollectionSchema, DataType, Collection, utility
from milvus_client import connect_milvus
import os
from dotenv import load_dotenv
load_dotenv()

def create_collections():
    connect_milvus()
    # Participants collection
    if not utility.has_collection("participants"):
        participant_fields = [
            FieldSchema(name="pid", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=os.getenv("EMB_DIM")),
        ]
        participants_schema = CollectionSchema(participant_fields, description="Participants collection")
        Collection(name="participants", schema=participants_schema)

    # Hackathons collection
    if not utility.has_collection("hackathons"):
        hackathon_fields = [
            FieldSchema(name="hid", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=os.getenv("EMB_DIM")),
        ]
        hackathon_schema = CollectionSchema(hackathon_fields, description="Hackathons collection")
        Collection(name="hackathons", schema=hackathon_schema)

create_collections()