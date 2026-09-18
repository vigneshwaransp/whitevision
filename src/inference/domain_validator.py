"""Out-of-Distribution (OOD) Domain Validator for Construction Vehicles.

Guards the closed-set 8-class construction classifier against unsupported images,
such as passenger cars, sports cars, domestic animals, and everyday non-construction objects.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

logger = logging.getLogger("DomainValidator")

# Pretrained ImageNet classes representing passenger / consumer personal transport
PASSENGER_VEHICLE_CLASSES = {
    "sports_car", "racer", "convertible", "cab", "beach_wagon", "minivan",
    "limousine", "jeep", "Model_T", "golfcart", "go-kart",
    "motorcycle", "moped", "motor_scooter", "bicycle", "mountain_bike", "tricycle",
    "recreational_vehicle", "minibus", "station_wagon", "kart"
}

# ImageNet classes representing heavy machinery, commercial trucks, and industrial equipment
CONSTRUCTION_AND_INDUSTRIAL_CLASSES = {
    "crane", "tractor", "harvester", "thresher", "forklift", "tow_truck",
    "garbage_truck", "snowplow", "trailer_truck", "steam_shovel", "plow",
    "lumbermill", "tank", "half_track", "moving_van", "fire_engine",
    "drilling_platform", "steel_arch_bridge", "lawn_mower"
}

_GLOBAL_IMAGENET_MODEL = None
_CENTROIDS = None
_FEATURE_MODEL = None


def get_imagenet_model():
    """Lazily load or retrieve the cached EfficientNetB0 ImageNet backbone."""
    global _GLOBAL_IMAGENET_MODEL
    if _GLOBAL_IMAGENET_MODEL is None:
        from tensorflow.keras.applications.efficientnet import EfficientNetB0
        logger.info("Initializing ImageNet domain gatekeeper backbone...")
        _GLOBAL_IMAGENET_MODEL = EfficientNetB0(weights="imagenet")
    return _GLOBAL_IMAGENET_MODEL


def get_centroids():
    """Load precomputed training set centroids for the 8 construction classes."""
    global _CENTROIDS
    if _CENTROIDS is None:
        p = Path("models/centroids.npy")
        if p.exists():
            try:
                _CENTROIDS = np.load(p, allow_pickle=True).item()
            except Exception as e:
                logger.warning(f"Could not load centroids: {e}")
    return _CENTROIDS


def get_feature_model():
    """Load submodel up to feature pooling layer for geometric distance checks."""
    global _FEATURE_MODEL
    if _FEATURE_MODEL is None:
        from tensorflow import keras
        mp = Path("models/construction_vehicle_efficientnetb0.keras")
        fallback_s1 = Path("models/stage1_feature_extractor.keras")
        target_mp = mp if mp.exists() else fallback_s1
        if target_mp.exists():
            try:
                full_model = keras.models.load_model(target_mp)
                _FEATURE_MODEL = keras.Model(inputs=full_model.input, outputs=full_model.layers[-2].output)
            except Exception as e:
                logger.warning(f"Could not build feature extractor: {e}")
    return _FEATURE_MODEL


def validate_construction_domain(
    image_input: Union[str, Path, Image.Image],
    imagenet_model=None,
    target_size: int = 224,
) -> Dict:
    """Validate whether an input image belongs to the construction vehicle domain.

    Returns:
        dict with keys:
            - is_supported (bool): True if candidate construction vehicle, False if unsupported.
            - detected_category (str): Human-readable name of detected primary entity.
            - error_message (str or None): User-facing error message when unsupported.
            - top_predictions (list): Top ImageNet labels with confidence scores.
    """
    from tensorflow.keras.applications.efficientnet import decode_predictions, preprocess_input
    from src.inference.predict import letterbox_image

    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")

    model = imagenet_model if imagenet_model is not None else get_imagenet_model()

    # Preprocess with standard letterboxing
    letterboxed = letterbox_image(img, target_size=target_size)
    arr = np.array(letterboxed, dtype=np.float32)
    prep = preprocess_input(np.expand_dims(arr, axis=0))

    raw_preds = model.predict(prep, verbose=0)
    top_decoded = decode_predictions(raw_preds, top=5)[0]

    top_label = top_decoded[0][1]
    top_prob = float(top_decoded[0][2])

    top_summary = [
        {"class": item[1], "confidence": round(float(item[2]), 4), "percentage": f"{item[2]*100:.1f}%"}
        for item in top_decoded
    ]

    # Calculate cumulative confidence for passenger vehicles
    car_conf = sum(float(item[2]) for item in top_decoded if item[1] in PASSENGER_VEHICLE_CLASSES)
    is_industrial_candidate = any(item[1] in CONSTRUCTION_AND_INDUSTRIAL_CLASSES for item in top_decoded)

    # Check 1: Passenger car / sports car / taxi / consumer personal vehicle
    # Must reject if top prediction is a passenger vehicle, or car confidence is high and no heavy equipment detected
    if (top_label in PASSENGER_VEHICLE_CLASSES) or (car_conf >= 0.35 and not is_industrial_candidate):
        clean_name = top_label.replace("_", " ").title()
        detected_entity = f"Passenger Vehicle / {clean_name}"
        error_msg = (
            f"Unsupported image: A {clean_name.lower()} was detected ({top_prob * 100:.1f}% confidence). "
            f"This system only classifies heavy construction-site machinery (Bulldozer, Dump Truck, "
            f"Excavator, Grader, Loader, Mixer Truck, Mobile Crane, Roller). "
            f"Passenger cars and personal vehicles are not supported."
        )
        return {
            "is_supported": False,
            "detected_category": detected_entity,
            "error_message": error_msg,
            "top_predictions": top_summary,
            "primary_label": top_label,
            "primary_confidence": round(top_prob, 4),
        }

    # Check 2: Non-construction objects, media artwork, humans, apparel, and everyday items
    # If no heavy equipment or industrial machinery appears in the top predictions and the top entity exceeds 18%, reject.
    if not is_industrial_candidate and top_prob >= 0.18:
        clean_name = top_label.replace("_", " ").title()
        detected_entity = f"Non-Construction Entity / {clean_name}"
        error_msg = (
            f"Unsupported image: No construction-site vehicle detected. Identified subject: {clean_name} "
            f"({top_prob * 100:.1f}% confidence). This system only classifies heavy construction machinery: "
            f"Bulldozer, Dump Truck, Excavator, Grader, Loader, Mixer Truck, Mobile Crane, Roller."
        )
        return {
            "is_supported": False,
            "detected_category": detected_entity,
            "error_message": error_msg,
            "top_predictions": top_summary,
            "primary_label": top_label,
            "primary_confidence": round(top_prob, 4),
        }

    # Check 3: Deep Feature Space Centroid Distance Gate (Geometric Manifold Alignment)
    centroids = get_centroids()
    feat_model = get_feature_model()
    if centroids is not None and feat_model is not None:
        try:
            feats = feat_model.predict(prep, verbose=0)[0]
            f_norm = np.linalg.norm(feats)
            if f_norm > 0:
                feat_vec = feats / f_norm
                sims = [float(np.dot(feat_vec, c_vec)) for c_vec in centroids.values()]
                max_sim = max(sims)
                # If alignment to all 8 construction class centroids is below 0.12, reject
                if max_sim < 0.12:
                    return {
                        "is_supported": False,
                        "detected_category": "Out-of-Distribution Image",
                        "error_message": (
                            f"Unsupported image: No construction-site vehicle detected (geometric alignment score: {max_sim:.2f} < 0.12). "
                            f"This system strictly classifies heavy construction machinery: Bulldozer, Dump Truck, "
                            f"Excavator, Grader, Loader, Mixer Truck, Mobile Crane, Roller."
                        ),
                        "top_predictions": top_summary,
                        "primary_label": top_label,
                        "primary_confidence": round(top_prob, 4),
                        "domain_alignment": round(max_sim, 4),
                    }
        except Exception as e:
            logger.debug(f"Centroid gate check skipped: {e}")

    # Otherwise, image is a valid construction domain candidate
    clean_top = top_label.replace("_", " ").title()
    return {
        "is_supported": True,
        "detected_category": f"Construction Domain ({clean_top})",
        "error_message": None,
        "top_predictions": top_summary,
        "primary_label": top_label,
        "primary_confidence": round(top_prob, 4),
    }
