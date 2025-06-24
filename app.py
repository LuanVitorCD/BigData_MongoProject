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

# FastAPI app
app = FastAPI(
    title="Model Cache IA",
    description="""
    Esta é uma API simples de demonstração para um sistema de cache de modelos preditivos com MongoDB.

    A API permite enviar um perfil de usuário (como 'novato' ou 'premium') e uma lista de features (valores numéricos) para que o modelo correspondente seja carregado automaticamente do MongoDB e uma predição seja realizada.

    ▶ Acesse o endpoint `/predict` via POST para testar.
    ▶ Use `/` para ver se a aplicação está online.
    """,
    version="1.0.0"
)

@app.on_event("startup")
def init_db():
    try:
        if col.count_documents({}) == 0:
            from sklearn.linear_model import LinearRegression
            import numpy as np

            X = np.array([[1, 2, 3], [2, 1, 0]])
            y = np.array([1.8, 3.2])
            model = LinearRegression().fit(X, y)

            col.insert_one({
                "_id": "modelo_ml",
                "weights": model.coef_.tolist(),
                "bias": model.intercept_.item()
            })
            print("✅ Modelo treinado e inserido com sucesso")
    except Exception as e:
        print(f"Erro ao conectar ou inserir no MongoDB: {e}")

@app.get("/")
def root():
    return {
        "status": "online",
        "description": "Este projeto demonstra um cache de modelos preditivos simples em MongoDB. Envie uma requisição POST para /predict com um perfil (novato ou premium) e uma lista de features para receber uma predição. Veja mais em /docs."
    }

class Request(BaseModel):
    profile: str
    features: list[float]

@app.post("/predict")
def predict(req: Request):
    model = col.find_one({"_id": req.profile})
    if not model:
        raise HTTPException(status_code=404, detail="Perfil não encontrado")
    import numpy as np
    weights = np.array(model["weights"])
    bias = model["bias"]
    features = np.array(req.features)
    result = float(np.dot(features, weights) + bias)
    return {"profile": req.profile, "prediction": result}