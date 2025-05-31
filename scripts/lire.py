import pickle

with open("embeddings/index.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data))
# Par exemple, si data est un tuple :
for i, d in enumerate(data):
    print(f"Element {i}: type={type(d)}")
    # Pour afficher un extrait (exemple)
    if hasattr(d, '__len__') and len(d) < 10:
        print(d)
print("Exemple d'élément metadata :", data[2][9001])
