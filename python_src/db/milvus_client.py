from pymilvus import connections

def connect_milvus():
    import os
    from dotenv import load_dotenv
    load_dotenv()

    connections.connect(
        alias="default",
        uri=os.getenv("MILVUS_URI"),
        token=os.getenv("MILVUS_TOKEN")
    )