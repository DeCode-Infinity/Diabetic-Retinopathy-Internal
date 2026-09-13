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
from skimage.morphology import skeletonize


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


def create_fov_mask(img_rgb: np.ndarray) -> np.ndarray:
    """
    Detect the valid retinal field-of-view so that the black camera
    background can never be classified as vessels.
    """

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Retinal photographs normally have a black border/background.
    # A low threshold isolates the illuminated retinal field.
    mask = (gray > 12).astype(np.uint8) * 255

    # Close small gaps around the retinal boundary.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (21, 21)
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Keep only the largest connected region: the actual fundus.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if num_labels > 1:
        largest_label = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )
        mask = np.where(
            labels == largest_label,
            255,
            0
        ).astype(np.uint8)

    # Erode the boundary slightly. Vesselness is unreliable close to the
    # circular camera edge and should not be drawn there.
    mask = cv2.erode(
        mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 9)
        ),
        iterations=1
    )

    return mask


def compute_frangi_vesselness(
    enhanced_gray: np.ndarray,
    fov_mask: np.ndarray,
    sigmas=(1.0, 2.0, 3.0, 4.0),
    beta: float = 0.5
):
    """
    Multiscale Hessian vesselness tuned for an actual retinal vessel
    overlay. The important difference from the previous version is that
    the Frangi response is normalized per scale and c is derived from the
    Hessian response instead of using a fixed value that suppresses nearly
    all responses on images normalized to 0..1.
    """

    img_float = enhanced_gray.astype(np.float32) / 255.0
    vesselness = np.zeros_like(img_float, dtype=np.float32)

    for sigma in sigmas:
        H_elems = hessian_matrix(
            img_float,
            sigma=sigma,
            order="rc"
        )
        lambda1, lambda2 = hessian_matrix_eigvals(H_elems)

        # skimage returns eigenvalues ordered by absolute magnitude.
        abs_l1 = np.abs(lambda1)
        abs_l2 = np.abs(lambda2)

        rb2 = (abs_l1 / (abs_l2 + 1e-7)) ** 2
        s2 = abs_l1 ** 2 + abs_l2 ** 2

        # Automatic structureness scale. A fixed c=15 is far too large
        # when the image itself has already been normalized to 0..1.
        valid_s = np.sqrt(s2[fov_mask > 0])
        valid_s = valid_s[np.isfinite(valid_s)]

        if valid_s.size == 0:
            continue

        c = max(
            float(np.percentile(valid_s, 90)) * 0.5,
            1e-5
        )

        vessel_sigma = (
            np.exp(-rb2 / (2.0 * beta ** 2))
            *
            (
                1.0
                -
                np.exp(-s2 / (2.0 * c ** 2))
            )
        )

        vessel_sigma[fov_mask == 0] = 0

        # Normalize each scale before taking the maximum. Otherwise one
        # numerical scale can dominate every other vessel width.
        valid = vessel_sigma[fov_mask > 0]
        if valid.size > 0:
            lo, hi = np.percentile(valid, [1, 99.5])
            vessel_sigma = np.clip(
                (vessel_sigma - lo) / (hi - lo + 1e-8),
                0,
                1
            )

        vesselness = np.maximum(
            vesselness,
            vessel_sigma.astype(np.float32)
        )

    vesselness[fov_mask == 0] = 0

    vesselness_norm = (
        np.clip(vesselness, 0, 1) * 255
    ).astype(np.uint8)

    # Frangi response alone can be weak around very fine vessels.
    # Build a classical dark-line response from the same enhanced image
    # and use it only as a support signal for the Frangi tracing.
    background = cv2.GaussianBlur(
        enhanced_gray,
        (0, 0),
        5.0
    )
    dark_line = cv2.subtract(
        background,
        enhanced_gray
    ).astype(np.float32)

    dark_line[fov_mask == 0] = 0

    valid_dark = dark_line[fov_mask > 0]
    if valid_dark.size > 0:
        dlo, dhi = np.percentile(
            valid_dark,
            [5, 99]
        )
        dark_line = np.clip(
            (dark_line - dlo) / (dhi - dlo + 1e-8),
            0,
            1
        )
    else:
        dark_line = np.zeros_like(
            vesselness,
            dtype=np.float32
        )

    # Frangi remains the dominant signal, while the dark-line response
    # reconnects obvious vessels that have a weak Hessian response.
    combined = (
        0.75 * vesselness
        +
        0.25 * dark_line
    )
    combined[fov_mask == 0] = 0

    valid_combined = combined[fov_mask > 0]

    if valid_combined.size == 0:
        empty = np.zeros_like(
            enhanced_gray,
            dtype=np.uint8
        )
        return vesselness_norm, empty, 0.0

    # Keep a relatively broad candidate set. Skeletonization will turn
    # it into thin vessel lines, so we do not need an aggressively high
    # threshold that makes the overlay disappear.
    threshold = float(
        np.percentile(
            valid_combined,
            76
        )
    )

    vessel_mask = (
        combined >= threshold
    ).astype(np.uint8)

    vessel_mask = cv2.bitwise_and(
        vessel_mask,
        (fov_mask > 0).astype(np.uint8)
    )

    # A tiny median filter removes isolated points without deleting thin
    # vessels, unlike the previous morphological opening.
    vessel_mask = cv2.medianBlur(
        vessel_mask * 255,
        3
    )

    # Skeletonize before component filtering so long, thin vessels are
    # represented by their actual geometry instead of their filled area.
    skeleton = skeletonize(
        vessel_mask > 0
    ).astype(np.uint8)

    # Remove only tiny skeleton fragments. Long vessels survive even when
    # their original filled region is very narrow.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        skeleton,
        connectivity=8
    )

    cleaned = np.zeros_like(
        skeleton,
        dtype=np.uint8
    )

    min_component_pixels = max(
        12,
        int(np.sqrt(enhanced_gray.size) * 0.08)
    )

    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_component_pixels:
            cleaned[labels == label] = 1

    # One-pixel skeletons can disappear when the frontend downsizes the
    # base64 image. Dilate once to create a crisp but still narrow trace.
    vessel_mask = cv2.dilate(
        cleaned * 255,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        ),
        iterations=1
    )

    vessel_mask = cv2.bitwise_and(
        vessel_mask,
        fov_mask
    )

    fov_pixels = max(
        int(np.count_nonzero(fov_mask)),
        1
    )

    vessel_pixels = int(
        np.count_nonzero(vessel_mask)
    )

    vessel_density = float(
        vessel_pixels / fov_pixels
    )

    return (
        vesselness_norm,
        vessel_mask,
        vessel_density
    )

def frangi_vesselness(
    img_rgb: np.ndarray
):
    """
    Generate a clean morphology-style Frangi vessel overlay:
    grayscale/CLAHE retinal image + thin cyan vessel tracing.
    """

    bgr_img = cv2.cvtColor(
        img_rgb,
        cv2.COLOR_RGB2BGR
    )

    # 1. Green channel denoising.
    green_denoised = (
        isolate_and_denoise_green_channel(
            bgr_img
        )
    )

    # 2. Retinal illumination normalization.
    normalized_img = (
        illumination_normalization(
            green_denoised
        )
    )

    # 3. CLAHE for vessel contrast.
    enhanced_gray = (
        apply_vessel_clahe(
            normalized_img,
            clip_limit=2.0,
            tile_size=(8, 8)
        )
    )

    # 4. Valid retinal field-of-view.
    fov_mask = create_fov_mask(
        img_rgb
    )

    # 5. Clean multiscale Frangi vessel extraction.
    (
        vesselness_norm,
        vessel_mask,
        vessel_density
    ) = compute_frangi_vesselness(
        enhanced_gray,
        fov_mask
    )

    # Use the enhanced grayscale image as the background, matching the
    # morphology-style output shown in the reference.
    base_rgb = cv2.cvtColor(
        enhanced_gray,
        cv2.COLOR_GRAY2RGB
    )

    # Keep the invalid camera background dark.
    base_rgb[fov_mask == 0] = 0

    composite = base_rgb.astype(
        np.float32
    )

    # Thin cyan vessel tracing.
    cyan = np.array(
        [35, 205, 225],
        dtype=np.float32
    )

    alpha = 0.82

    vessel_pixels = vessel_mask > 0

    composite[vessel_pixels] = (
        composite[vessel_pixels]
        *
        (1.0 - alpha)
        +
        cyan
        *
        alpha
    )

    composite = np.clip(
        composite,
        0,
        255
    ).astype(np.uint8)

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
