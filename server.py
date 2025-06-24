from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import predict
import uvicorn

class Request(BaseModel):
    profile: str
    features: list[float]

app = FastAPI()

@app.post("/predict")
def api_predict(req: Request):
    try:
        value = predict(req.features, req.profile)
        return {"profile": req.profile, "prediction": value}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
