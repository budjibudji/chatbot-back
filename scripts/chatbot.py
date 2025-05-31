import faiss
import pickle
from sentence_transformers import SentenceTransformer
import requests
import time  # ➤ Pour mesurer le temps

# Charger le modèle d'encodage de phrases
model = SentenceTransformer("all-MiniLM-L6-v2")

# Charger l'index FAISS et les métadonnées
with open("embeddings/index.pkl", "rb") as f:
    index, embeddings, metadatas = pickle.load(f)

def recherche(query, top_k=5):
    # Encoder la question
    vec = model.encode([query])
    # Recherche des top_k offres proches
    scores, ids = index.search(vec, top_k)
    # Retourner les métadatas (offres) correspondantes
    return [metadatas[i] for i in ids[0]]

def interroger_mistral(query, docs):
    # Construire un contexte uniquement avec les descriptions
    context = "\n\n---\n\n".join([
        f"Titre: {d.get('title','')}\nLieu: {d.get('location','')}\nDescription: {d.get('description','')}\nURL: {d.get('url','')}"
        for d in docs
    ])
    
    prompt = f"""
Tu es un expert en ressources humaines et data science.

Voici plusieurs descriptions d'offres d'emploi sélectionnées comme étant les plus proches de la question posée :
{context}

À partir de ces descriptions uniquement, 

Question :  
{query}
"""

    # Mesure du temps de début
    start_time = time.time()

    # Appel à l'API locale de Mistral
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    })

    # Mesure du temps de fin
    end_time = time.time()

    # Temps écoulé en secondes
    elapsed_time = end_time - start_time

    # Affichage du temps
    print(f"\n⏱ Temps de réponse : {elapsed_time:.2f} secondes\n")

    return response.json().get("response", "Pas de réponse.")

if __name__ == "__main__":
    query = "roadmap pour etre data scientist au maroc?"
    docs = recherche(query)

    réponse = interroger_mistral(query, docs)
    print("\n🤖 Réponse du chatbot :\n")
    print(réponse)
