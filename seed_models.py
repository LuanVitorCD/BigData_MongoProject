from pymongo import MongoClient
from config import MONGODB_URI, DB_NAME, COLLECTION_NAME

# Conexão
client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
col = db[COLLECTION_NAME]

# Exemplos de modelos (pesos + bias)
models = [
    {"_id": "novato",  "weights": [0.1, 0.2, 0.3], "bias": 0.5},
    {"_id": "premium", "weights": [0.4, 0.5, 0.6], "bias": 1.0}
]

# Insere ou substitui
for m in models:
    col.replace_one({"_id": m["_id"]}, m, upsert=True)

print("Modelos inseridos:")
for doc in col.find():
    print(doc)
