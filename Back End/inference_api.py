"""
DR Screen AI — Inference Backend

Loads your trained checkpoints/best_model.pth (from train_classifier.py) and
serves real predictions to the React frontend (dr-screen-ai.jsx).

No random numbers anywhere in this file — grade, confidence, probabilities,
quality metrics, the CLAHE-enhanced image, and the Grad-CAM heatmap are all
computed from the actual uploaded image and the actual trained model.

Run:
    pip install fastapi uvicorn python-multipart timm torch torchvision \
                opencv-python-headless pillow pytorch-grad-cam
    uvicorn inference_api:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import base64
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import timm
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ─────────────────────────── Config ───────────────────────────
MODEL_PATH = "checkpoints/best_model.pth"   # from train_classifier.py
MODEL_NAME = "efficientnet_b4"
IMAGE_SIZE = 380
NUM_CLASSES = 5
QUALITY_REJECT_THRESHOLD = 50   # below this, ask for a recapture instead of grading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="DR Screen AI — Inference API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ─────────────────────────── Load model once at startup ───────────────────────────
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval().to(device)
print(f"Loaded {MODEL_NAME} from {MODEL_PATH} on {device}")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────── Helpers ───────────────────────────
def encode_jpeg_b64(img_rgb_uint8: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR))
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

def quality_metrics(gray: np.ndarray):
    """Real Shannon entropy + Laplacian sharpness — same formulas used in
    the preprocessing scripts, not simulated numbers."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    hist_norm = hist_norm[hist_norm > 0]
    entropy = float(-np.sum(hist_norm * np.log2(hist_norm)))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    quality_score = int(np.clip((entropy / 8.0) * 60 + (min(sharpness, 300) / 300) * 40, 0, 100))
    return quality_score, sharpness, entropy

def clahe_enhance(img_rgb: np.ndarray) -> np.ndarray:
    """Real CLAHE on the green channel — same technique from the
    preprocessing pipeline, actually applied here (not a CSS filter)."""
    green = img_rgb[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    return cv2.merge([enhanced, enhanced, enhanced])

# ─────────────────────────── Routes ───────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    img_rgb = np.array(pil_img)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    quality, sharpness, entropy = quality_metrics(gray)

    # Reject ungradeable images with actionable recapture feedback, instead
    # of silently grading a poor-quality photo — this is what the problem
    # statement calls for in Phase 1.
    if quality < QUALITY_REJECT_THRESHOLD:
        issues = []
        if entropy < 5.0:
            issues.append("Image too dark or overexposed")
        if sharpness < 80:
            issues.append("Image too blurry — hold the camera steady")
        if not issues:
            issues.append("Overall image quality too low to grade reliably")
        raise HTTPException(status_code=422, detail={
            "message": "Image not gradeable — please recapture",
            "issues": issues,
            "quality": quality,
        })

    enhanced_rgb = clahe_enhance(img_rgb)

    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # ── real model prediction ──
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    grade = int(np.argmax(probs))
    confidence = float(probs[grade])

    # ── real Grad-CAM on the actual trained classifier ──
    target_layers = [model.conv_head]   # last conv layer before pooling, in timm's EfficientNet
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(grade)])[0]
    rgb_for_cam = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
    heatmap_rgb = show_cam_on_image(rgb_for_cam, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET)

    return {
        "grade": grade,
        "confidence": confidence,
        "probs": probs.tolist(),
        "quality": quality,
        "sharpness": f"{sharpness:.0f}",
        "entropy": f"{entropy:.2f}",
        "enhanced": encode_jpeg_b64(enhanced_rgb),
        "heatmap": encode_jpeg_b64(heatmap_rgb),
    }
