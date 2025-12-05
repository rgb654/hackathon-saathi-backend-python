from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")

def get_skill_embedding(skills):
    """
    Given a list of skills, return a single aggregated embedding vector.
    """
    input_text = ", ".join(skills)  # e.g., "Python, PyTorch, FastAPI"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the [CLS] token representation
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (1, hidden_size)
    embedding = cls_embedding.squeeze().numpy()  # shape: (hidden_size,)

    # Normalize the embedding (L2 norm)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        normalized_embedding = embedding
    else:
        normalized_embedding = embedding / norm

    return normalized_embedding