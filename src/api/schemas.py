from pydantic import BaseModel, Field
from typing import List, Optional

class GenerateRequest(BaseModel):
    labels: List[int] = Field(..., min_length=1, max_length=100)
    seed: Optional[int] = 42

class GenerateResponse(BaseModel):
    labels: List[int]
    images_base64: List[str]
    model_name: str