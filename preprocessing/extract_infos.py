import pandas as pd
import requests
import json
from tqdm import tqdm
import time

# Chemins des fichiers
input_csv = "data/offres.csv"
output_csv = "data/offres_enrichies.csv"

# Charger les données
df = pd.read_csv(input_csv)

# Colonnes à remplir
colonnes_a_remplir = ["salary", "job_type", "experience", "skills", "company_size", "industry"]
for col in colonnes_a_remplir:
    if col not in df.columns:
        df[col] = ""

# Appel à Ollama avec Mistral
def extraire_infos_avec_mistral(description):
    prompt = f"""
Tu es un assistant intelligent. Analyse la description suivante et retourne un JSON strictement valide avec les clés :
"salary", "job_type", "experience", "skills", "company_size", "industry".

Description :
\"\"\"{description}\"\"\"

Réponse attendue : un JSON uniquement, sans texte autour.
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.2
            }
        )
        output = response.json()["response"].strip()
        # Affichage de debug
        print("Réponse brute de Mistral:", output)

        # Tentative de parsing direct
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            # Correction simple : remplacer ' par "
            output_fixed = output.replace("'", '"')
            return json.loads(output_fixed)
    except Exception as e:
        print("Erreur Ollama:", e)
        return {}

# Sauvegarde progressive
with open(output_csv, "w", encoding="utf-8", newline="") as f_out:
    df.iloc[0:0].to_csv(f_out, index=False)  # En-tête CSV

    for i in tqdm(range(len(df)), desc="Remplissage en cours"):
        row = df.iloc[i]
        desc = row.get("description", "")
        if not isinstance(desc, str) or desc.strip() == "":
            continue

        infos = extraire_infos_avec_mistral(desc)

        for key in colonnes_a_remplir:
            if key in infos and (pd.isna(row[key]) or row[key] == ""):
                df.at[i, key] = infos[key]

        df.iloc[[i]].to_csv(f_out, mode="a", index=False, header=False)
        time.sleep(0.5)  # éviter de saturer Ollama
