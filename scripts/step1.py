# scripts/embed_offres.py
import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/offres.csv")
model = SentenceTransformer("all-MiniLM-L6-v2")

docs = df["description"].fillna("").tolist()
metadatas = df[["title", "location", "url", "description"]].fillna("").to_dict(orient="records")

embeddings = model.encode(docs, show_progress_bar=True)

# Index FAISS
index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(embeddings)

# Sauvegarde
with open("embeddings/index.pkl", "wb") as f:
    pickle.dump((index, embeddings, metadatas), f)
