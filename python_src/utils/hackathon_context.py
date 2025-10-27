from bson import ObjectId
from db.mongo_client import get_hackathons_collection
from models.hackathon import HackathonDreamTeam
import json

from google import genai
# import google.generativeai as genai

import os
from dotenv import load_dotenv

load_dotenv()


def flatten_document(doc, parent_key="", sep="."):
    """Recursively flattens nested dicts/lists into a string key:value format."""
    items = []
    for k, v in doc.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.append(flatten_document(v, new_key, sep=sep))
        elif isinstance(v, list):
            list_items = []
            for i, val in enumerate(v):
                if isinstance(val, dict):
                    list_items.append(flatten_document(val, f"{new_key}[{i}]", sep=sep))
                else:
                    list_items.append(f"{new_key}[{i}]: {val}")
            items.append("\n".join(list_items))
        else:
            items.append(f"{new_key}: {v}")
    return "\n".join(items)

def generate_hackathon_context(hackathon_id: str) -> str:
    collection = get_hackathons_collection()
    doc = collection.find_one({"_id": ObjectId(hackathon_id)})

    if not doc:
        return f"No hackathon found with _id {hackathon_id}"

    # Convert ObjectId/Date types to JSON serializable
    def bson_default(obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return str(obj)

    clean_doc = json.loads(json.dumps(doc, default=bson_default))

    return flatten_document(clean_doc)


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_dream_team_skills(hackathon_context: str) -> HackathonDreamTeam:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
        You are given details about a hackathon:

        {hackathon_context}

        Based on this, generate a list of 10-20 technical skills that the 'dream team' should have to maximize their chances of winning this hackathon. Return only the structured JSON.
        """,
        config={
            "response_mime_type": "application/json",
            "response_schema": HackathonDreamTeam,
        },
    )

    return {'hackathon_name':response.parsed.hackathon_name, 'target_skills':response.parsed.required_skills}

def generate_hackathon_skills(hidx):
    context_string = generate_hackathon_context(hidx)
    return generate_dream_team_skills(context_string)