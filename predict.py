from pymongo import MongoClient
from config import MONGODB_URI, DB_NAME, COLLECTION_NAME

def predict(features: list[float], profile: str) -> float:
    client = MongoClient(MONGODB_URI)
    col = client[DB_NAME][COLLECTION_NAME]
    model = col.find_one({"_id": profile})
    if not model:
        raise ValueError(f"Perfil '{profile}' não encontrado no cache.")
    weights, bias = model["weights"], model["bias"]
    return sum(f * w for f, w in zip(features, weights)) + bias

if __name__ == "__main__":
    p = input("Perfil (novato/premium): ")
    feats = list(map(float, input("Features (vírgula): ").split(',')))
    result = predict(feats, p)
    print(f"Predição [{p}]: {result}")
