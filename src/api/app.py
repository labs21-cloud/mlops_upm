from fastapi import FastAPI, HTTPException

from src.api.schemas import GenerateRequest, GenerateResponse
from src.api.service import CVAEGeneratorService

app = FastAPI(
    title="CVAE MNIST API",
    description="API para generación condicionada de dígitos MNIST",
    version="1.0.0",
)

service = CVAEGeneratorService()

@app.get("/health")
def health():
    return service.health()

@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    if any(label < 0 or label > 9 for label in request.labels):
        raise HTTPException(status_code=400, detail="All labels must be between 0 and 9.")

    images_base64 = service.generate(labels=request.labels, seed=request.seed)

    return GenerateResponse(
        labels=request.labels,
        images_base64=images_base64,
        model_name="cvae-mnist",
    )