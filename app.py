import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv

# Load env
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "model_cache_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "models")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
col = db[COLLECTION_NAME]

# Popular o banco se vazio
if col.count_documents({}) == 0:
    col.insert_many([
        {"_id": "novato",  "weights": [0.1, 0.2, 0.3], "bias": 0.5},
        {"_id": "premium", "weights": [0.4, 0.5, 0.6], "bias": 1.0}
    ])

# FastAPI app
app = FastAPI()

@app.get("/")
def root():
    return {"status": "online"}

class Request(BaseModel):
    profile: str
    features: list[float]

@app.post("/predict")
def predict(req: Request):
    model = col.find_one({"_id": req.profile})
    if not model:
        raise HTTPException(status_code=404, detail="Perfil não encontrado")
    result = sum(f*w for f, w in zip(req.features, model["weights"])) + model["bias"]
    return {"profile": req.profile, "prediction": result}