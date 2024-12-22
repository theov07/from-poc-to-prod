from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
from run import TextPredictionModel
import uvicorn
from fastapi.openapi.utils import get_openapi
from fastapi.responses import RedirectResponse
import os


app = FastAPI()


artefacts_url = "train/data/artefacts"

host = "127.0.0.1"


class PredictionRequest(BaseModel):
    text: str
    top_k: int
    date_artefacts_url: str  # 2024-12-11-17-02-59 FOR TESTING



@app.get("/docs", include_in_schema=False)
def custom_swagger_ui_html():
    return get_openapi(title="Custom API", version="1.0.0", routes=app.routes)



@app.get("/")
def index():
    return RedirectResponse(url="/docs")



@app.post("/predict")
def get_text(request: PredictionRequest):
    try:
        artefacts_url = os.path.join("train/data/artefacts", request.date_artefacts_url)
        model = TextPredictionModel.from_artefacts(artefacts_url)
        predictions = model.predict(request.text, request.top_k)
        return {"text": request.text, "predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host=host, port=8000)