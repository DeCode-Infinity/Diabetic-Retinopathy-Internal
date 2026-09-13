"""
DR Screen AI — Inference Backend

Loads your trained checkpoints/best_model.pth and serves real predictions
to the React frontend.

Run:
    pip install fastapi uvicorn python-multipart timm torch torchvision \
                opencv-python-headless pillow pytorch-grad-cam scikit-image
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

from skimage.feature import hessian_matrix, hessian_matrix_eigvals


# ─────────────────────────── Config ───────────────────────────

MODEL_PATH = "checkpoints/best_model.pth"
MODEL_NAME = "efficientnet_b4"
IMAGE_SIZE = 380
NUM_CLASSES = 5
QUALITY_REJECT_THRESHOLD = 30

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ─────────────────────────── FastAPI App ───────────────────────────

app = FastAPI(
    title="DR Screen AI — Inference API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ─────────────────────────── Load Model ───────────────────────────

model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=NUM_CLASSES
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(state_dict)
model.eval().to(device)

print(
    f"Loaded {MODEL_NAME} "
    f"from {MODEL_PATH} "
    f"on {device}"
)


transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ─────────────────────────── Helpers ───────────────────────────

def encode_jpeg_b64(
    img_rgb_uint8: np.ndarray
) -> str:

    ok, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(
            img_rgb_uint8,
            cv2.COLOR_RGB2BGR
        )
    )

    if not ok:
        raise RuntimeError(
            "Could not encode image"
        )

    return (
        "data:image/jpeg;base64,"
        + base64.b64encode(buf).decode(
            "utf-8"
        )
    )


def quality_metrics(
    gray: np.ndarray
):
    """
    Real Shannon entropy and Laplacian
    sharpness measurements.
    """

    hist = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    hist_norm = (
        hist.ravel()
        / hist.sum()
    )

    hist_norm = hist_norm[
        hist_norm > 0
    ]

    entropy = float(
        -np.sum(
            hist_norm
            * np.log2(hist_norm)
        )
    )

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    quality_score = int(
        np.clip(
            (entropy / 8.0) * 60
            +
            (
                min(
                    sharpness,
                    300
                )
                / 300
            )
            * 40,
            0,
            100
        )
    )

    return (
        quality_score,
        sharpness,
        entropy
    )


def clahe_enhance(
    img_rgb: np.ndarray
) -> np.ndarray:
    """
    CLAHE enhancement on the green
    channel.
    """

    green = img_rgb[:, :, 1]

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        green
    )

    return cv2.merge([
        enhanced,
        enhanced,
        enhanced
    ])


# ─────────────────────────── Morphology Pipeline ───────────────────────────

def isolate_and_denoise_green_channel(
    bgr_img: np.ndarray
) -> np.ndarray:
    """
    Extract the retinal green channel
    and suppress high-frequency noise.
    """

    green = bgr_img[:, :, 1]

    denoised = cv2.GaussianBlur(
        green,
        (5, 5),
        1.0
    )

    return denoised


def illumination_normalization(
    green_denoised: np.ndarray
) -> np.ndarray:
    """
    Normalize non-uniform retinal
    illumination using large-kernel
    background subtraction.
    """

    background = cv2.GaussianBlur(
        green_denoised,
        (69, 69),
        30.0
    )

    normalized = cv2.addWeighted(
        green_denoised,
        1.0,
        background,
        -1.0,
        128.0
    )

    return np.clip(
        normalized,
        0,
        255
    ).astype(np.uint8)


def apply_vessel_clahe(
    normalized_img: np.ndarray,
    clip_limit: float = 2.0,
    tile_size=(8, 8)
) -> np.ndarray:
    """
    CLAHE preprocessing specifically
    for vessel extraction.
    """

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_size
    )

    return clahe.apply(
        normalized_img
    )


def compute_frangi_vesselness(
    enhanced_gray: np.ndarray,
    sigmas=[1.0, 2.0, 3.0],
    beta: float = 0.5,
    c: float = 15.0
):
    """
    Multiscale Hessian-based Frangi
    vesselness extraction.

    Returns:
        vesselness_norm
        binary_mask
        vessel_density
    """

    img_float = (
        enhanced_gray.astype(
            np.float32
        )
        / 255.0
    )

    vesselness = np.zeros_like(
        img_float
    )

    for sigma in sigmas:

        # Hessian matrix at the
        # current scale.
        H_elems = hessian_matrix(
            img_float,
            sigma=sigma,
            order="rc"
        )

        # Hessian eigenvalues.
        lambda1, lambda2 = (
            hessian_matrix_eigvals(
                H_elems
            )
        )

        abs_l1 = np.abs(
            lambda1
        )

        abs_l2 = np.abs(
            lambda2
        )

        # Blobness measure.
        rb = (
            abs_l1
            /
            (abs_l2 + 1e-5)
        ) ** 2

        # Structureness measure.
        s2 = (
            abs_l1 ** 2
            +
            abs_l2 ** 2
        )

        # 2D Frangi vesselness.
        vessel_sigma = (
            np.exp(
                -rb
                /
                (
                    2
                    *
                    (
                        beta ** 2
                    )
                )
            )
            *
            (
                1.0
                -
                np.exp(
                    -s2
                    /
                    (
                        2
                        *
                        (
                            c ** 2
                        )
                    )
                )
            )
        )

        # Maximum response across
        # all scales.
        vesselness = np.maximum(
            vesselness,
            vessel_sigma
        )

    # Normalize vessel response.
    vesselness_norm = cv2.normalize(
        vesselness,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Otsu thresholding.
    _, binary_mask = cv2.threshold(
        vesselness_norm,
        0,
        255,
        cv2.THRESH_BINARY
        +
        cv2.THRESH_OTSU
    )

    # Vascular density index.
    total_pixels = (
        enhanced_gray.size
    )

    vessel_pixels = (
        np.count_nonzero(
            binary_mask
        )
    )

    vessel_density = float(
        vessel_pixels
        /
        max(
            total_pixels,
            1
        )
    )

    return (
        vesselness_norm,
        binary_mask,
        vessel_density
    )


def frangi_vesselness(
    img_rgb: np.ndarray
):
    """
    Complete morphology.py-based
    retinal vessel extraction pipeline.

    Returns:
        composite RGB visualization
        vessel density
    """

    # Morphology pipeline expects
    # OpenCV BGR input.
    bgr_img = cv2.cvtColor(
        img_rgb,
        cv2.COLOR_RGB2BGR
    )

    # Step 1:
    # Green channel extraction and
    # denoising.
    green_denoised = (
        isolate_and_denoise_green_channel(
            bgr_img
        )
    )

    # Step 2:
    # Illumination normalization.
    normalized_img = (
        illumination_normalization(
            green_denoised
        )
    )

    # Step 3:
    # CLAHE enhancement.
    enhanced_gray = (
        apply_vessel_clahe(
            normalized_img,
            clip_limit=2.0,
            tile_size=(8, 8)
        )
    )

    # Step 4:
    # Multiscale Hessian-based
    # Frangi vessel extraction.
    (
        vesselness_norm,
        vessel_mask,
        vessel_density
    ) = compute_frangi_vesselness(
        enhanced_gray
    )

    # Base visualization.
    base_rgb = cv2.cvtColor(
        enhanced_gray,
        cv2.COLOR_GRAY2RGB
    ).astype(
        np.float32
    )

    # Cyan overlay.
    cyan_overlay = (
        np.zeros_like(
            base_rgb
        )
    )

    cyan_overlay[:, :, 1] = 255
    cyan_overlay[:, :, 2] = 255

    # Convert binary mask into an
    # alpha blending map.
    alpha = (
        vessel_mask.astype(
            np.float32
        )
        / 255.0
    ) * 0.75

    alpha_3ch = np.dstack([
        alpha,
        alpha,
        alpha
    ])

    composite = (
        base_rgb
        *
        (
            1.0
            -
            alpha_3ch
        )
        +
        cyan_overlay
        *
        alpha_3ch
    )

    composite = np.clip(
        composite,
        0,
        255
    ).astype(
        np.uint8
    )

    return (
        composite,
        vessel_density
    )


# ─────────────────────────── Routes ───────────────────────────

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "device": str(device)
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    raw = await file.read()

    try:
        pil_img = Image.open(
            io.BytesIO(raw)
        ).convert(
            "RGB"
        )

    except Exception:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not decode image"
            )
        )

    img_rgb = np.array(
        pil_img
    )

    gray = cv2.cvtColor(
        img_rgb,
        cv2.COLOR_RGB2GRAY
    )

    (
        quality,
        sharpness,
        entropy
    ) = quality_metrics(
        gray
    )

    # Reject poor quality images.
    if (
        quality
        <
        QUALITY_REJECT_THRESHOLD
    ):

        issues = []

        if entropy < 5.0:
            issues.append(
                "Image too dark or "
                "overexposed"
            )

        if sharpness < 80:
            issues.append(
                "Image too blurry — "
                "hold the camera steady"
            )

        if not issues:
            issues.append(
                "Overall image quality "
                "too low to grade reliably"
            )

        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Image not gradeable — "
                    "please recapture"
                ),
                "issues": issues,
                "quality": quality,
            }
        )

    # CLAHE visualization.
    enhanced_rgb = clahe_enhance(
        img_rgb
    )

    # Morphology-based vessel
    # visualization.
    (
        vessels_rgb,
        vessel_density
    ) = frangi_vesselness(
        img_rgb
    )

    # ── Real model prediction ──

    input_tensor = (
        transform(
            pil_img
        )
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        logits = model(
            input_tensor
        )

        probs = (
            F.softmax(
                logits,
                dim=1
            )
            .cpu()
            .numpy()[0]
        )

    grade = int(
        np.argmax(
            probs
        )
    )

    confidence = float(
        probs[grade]
    )

    # ── Real Grad-CAM ──

    target_layers = [
        model.conv_head
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[
            ClassifierOutputTarget(
                grade
            )
        ]
    )[0]

    rgb_for_cam = (
        cv2.resize(
            img_rgb,
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        )
        .astype(np.float32)
        /
        255.0
    )

    heatmap_rgb = (
        show_cam_on_image(
            rgb_for_cam,
            grayscale_cam,
            use_rgb=True,
            colormap=cv2.COLORMAP_JET
        )
    )

    return {
        "grade": grade,
        "confidence": confidence,
        "probs": probs.tolist(),

        "quality": quality,
        "sharpness": (
            f"{sharpness:.0f}"
        ),
        "entropy": (
            f"{entropy:.2f}"
        ),

        "vessel_density": round(
            vessel_density,
            6
        ),

        "enhanced": (
            encode_jpeg_b64(
                enhanced_rgb
            )
        ),

        "vessels": (
            encode_jpeg_b64(
                vessels_rgb
            )
        ),

        "heatmap": (
            encode_jpeg_b64(
                heatmap_rgb
            )
        ),
    }
