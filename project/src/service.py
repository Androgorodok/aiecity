from io import BytesIO
import json

import torch
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from torchvision import transforms

from src.models import create_model

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Corn Disease Classifier"
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

with open("artifacts/config.json") as f:
    config = json.load(f)

# Приводим class_names к списку, если это словарь
class_names = config["class_names"]
if isinstance(class_names, dict):
    class_names = list(class_names.values())
elif isinstance(class_names, list):
    class_names = class_names
else:
    class_names = [str(class_names)]

model = create_model(
    num_classes=len(class_names),
    pretrained=False,
)

model.load_state_dict(
    torch.load(
        "artifacts/best_model.pt",
        map_location=device,
    )
)

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


@app.get("/health")
def healthcheck():
    try:
        if model is None:
            raise ValueError("Model not loaded")
        
        logger.info("Healthcheck OK")
        return {"status": "ok"}
    
    except Exception as e:
        logger.error(f"Healthcheck FAILED: {e}")
        return {"status": "error", "message": str(e)}, 500


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    logger.info(f"Received file: {file.filename}")

    image = Image.open(
        BytesIO(
            await file.read()
        )
    ).convert("RGB")

    image = (
        transform(image)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        
        probs_list = probs[0].cpu().numpy().tolist()
        
        predictions = {}
        for i, class_name in enumerate(class_names):
            predictions[class_name] = float(probs_list[i])
        
        idx = probs.argmax().item()
        predicted_class = class_names[idx]
        confidence = float(probs[0][idx])

        # Лог с названием файла, предсказанием, уверенностью и ВСЕМИ классами
        probs_str = ", ".join([f"{k}={v:.16f}" for k, v in predictions.items()])
        logger.info(
            f"file={file.filename} | Predicted={predicted_class} confidence={confidence:.16f} | {probs_str}"
        )

    return {
        "predictions": predictions,
        "predicted_class": predicted_class,
        "confidence": confidence
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
