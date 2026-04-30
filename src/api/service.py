import base64
import io
from pathlib import Path

import torch
from PIL import Image

from src.models.cvae import CVAE
from src.inference.generate import generate_samples

MODEL_PATH = Path("models/cvae.ckpt")

class CVAEGeneratorService:
    def __init__(self, model_path: Path = MODEL_PATH, device: str | None = None):
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = CVAE.load_from_checkpoint(str(model_path), map_location=self.device)
        self.model.eval()
        self.model.to(self.device)

    def health(self) -> dict:
        return {
            "status": "ok",
            "model_loaded": self.model is not None,
            "device": str(self.device),
            "model_path": str(MODEL_PATH),
        }

    def generate(self, labels: list[int], seed: int | None = 42) -> list[str]:
        images = generate_samples(
            model=self.model,
            labels=labels,
            device=str(self.device),
            seed=seed,
        )

        encoded_images = []
        for img_tensor in images:
            img = (img_tensor.squeeze().numpy() * 255).astype("uint8")
            pil_img = Image.fromarray(img, mode="L")
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            encoded_images.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

        return encoded_images