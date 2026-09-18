"""Single and batch image inference engine with Top-K and confidence thresholding."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from PIL import Image
import yaml

from src.dataset.normalization import get_normalizer
from src.inference.domain_validator import validate_construction_domain

logger = logging.getLogger("Predictor")


def letterbox_image(img: Image.Image, target_size: int = 224, fill_color: Tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
    """Resize image preserving aspect ratio with centered neutral padding."""
    w, h = img.size
    if w == 0 or h == 0:
        return Image.new("RGB", (target_size, target_size), fill_color)
    scale = target_size / max(w, h)
    new_w = max(1, min(target_size, int(w * scale)))
    new_h = max(1, min(target_size, int(h * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_size, target_size), fill_color)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def center_crop_zoom(img: Image.Image, target_size: int = 224, zoom: float = 1.15) -> Image.Image:
    """Center zoom into core equipment features (useful for partial vehicle views)."""
    w, h = img.size
    crop_w = int(w / zoom)
    crop_h = int(h / zoom)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    cropped = img.crop((left, top, left + crop_w, top + crop_h))
    return letterbox_image(cropped, target_size=target_size)


class VehiclePredictor:
    """Production inference engine for construction-site vehicle classification."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        classes_path: Optional[Union[str, Path]] = None,
        config_path: str = "config/config.yaml",
        classes_config_path: str = "config/classes.yaml",
        confidence_threshold: Optional[float] = None,
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.threshold = confidence_threshold if confidence_threshold is not None else float(
            self.cfg.get("inference", {}).get("confidence_threshold", 0.50)
        )
        self.image_size = int(self.cfg.get("model", {}).get("image_size", 224))
        self.normalizer = get_normalizer(classes_config_path)

        # Resolve model path
        default_model = Path(self.cfg.get("export", {}).get("keras_model_path", "models/construction_vehicle_efficientnetb0.keras"))
        fallback_stage1 = Path("models/stage1_feature_extractor.keras")
        self.model_path = Path(model_path) if model_path else (default_model if default_model.exists() else fallback_stage1)

        # Resolve classes
        classes_json = Path(classes_path or self.cfg.get("export", {}).get("classes_json_path", "models/classes.json"))
        if classes_json.exists():
            with open(classes_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.classes = data.get("classes", self.normalizer.get_classes())
        else:
            self.classes = self.normalizer.get_classes()

        self._model = None

    @property
    def model(self):
        if self._model is None:
            from tensorflow import keras
            if not self.model_path.exists():
                raise FileNotFoundError(f"Trained model not found at {self.model_path}.")
            logger.info(f"Loading inference model from {self.model_path}...")
            self._model = keras.models.load_model(self.model_path)
        return self._model


    def preprocess(self, image_input: Union[str, Path, Image.Image], use_tta: bool = False) -> np.ndarray:
        """Preprocess input PIL image or file path using aspect-preserving letterboxing."""
        from PIL import ImageOps
        from tensorflow.keras.applications.efficientnet import preprocess_input

        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        if use_tta:
            # Multi-scale views: standard letterbox, center-zoom (partial view focus), and mirror
            v1 = letterbox_image(img, target_size=self.image_size)
            v2 = center_crop_zoom(img, target_size=self.image_size, zoom=1.15)
            v3 = ImageOps.mirror(v1)
            batch = np.stack([
                preprocess_input(np.array(v1, dtype=np.float32)),
                preprocess_input(np.array(v2, dtype=np.float32)),
                preprocess_input(np.array(v3, dtype=np.float32)),
            ])
            return batch

        letterboxed = letterbox_image(img, target_size=self.image_size)
        arr = np.array(letterboxed, dtype=np.float32)
        prep = preprocess_input(arr)
        return np.expand_dims(prep, axis=0)

    def predict(
        self,
        image_input: Union[str, Path, Image.Image],
        top_k: int = 3,
        threshold: Optional[float] = None,
        use_tta: bool = True,
        check_domain: bool = True,
    ) -> Dict:
        """Run classification on single image with domain validation, aspect-preserving TTA, and thresholding."""
        thresh = threshold if threshold is not None else self.threshold

        # Step 1: Out-of-Distribution Domain Gatekeeper
        if check_domain:
            domain_check = validate_construction_domain(image_input)
            if not domain_check["is_supported"]:
                return {
                    "is_supported": False,
                    "error": domain_check["error_message"],
                    "detected_object": domain_check["detected_category"],
                    "prediction": "unsupported",
                    "code": "ERR",
                    "display_name": "Unsupported Image",
                    "confidence": domain_check.get("primary_confidence", 0.0),
                    "percentage": f"{domain_check.get('primary_confidence', 0.0) * 100:.2f}%",
                    "status": "Rejected: Not a Construction Vehicle",
                    "threshold": thresh,
                    "is_confident": False,
                    "top_predictions": [],
                    "all_probabilities": {},
                    "domain_telemetry": domain_check,
                }

        batch = self.preprocess(image_input, use_tta=use_tta)

        raw_preds = self.model.predict(batch, verbose=0)
        if len(raw_preds.shape) == 2 and raw_preds.shape[0] > 1:
            # Average multi-view predictions for robust partial-image classification
            probs = np.mean(raw_preds, axis=0)
        else:
            probs = raw_preds[0] if len(raw_preds.shape) == 2 else raw_preds

        top_indices = np.argsort(probs)[::-1]

        best_idx = int(top_indices[0])
        best_prob = float(probs[best_idx])
        raw_pred_class = self.classes[best_idx]

        # Apply confidence threshold
        is_confident = best_prob >= thresh
        if is_confident:
            prediction_class = raw_pred_class
            display_prediction = self.normalizer.get_display_name(raw_pred_class)
            prediction_status = "High Confidence"
        else:
            prediction_class = "unknown"
            display_prediction = "Unknown / Low Confidence"
            prediction_status = f"Below Threshold ({thresh * 100:.1f}%)"

        # Build Top-K ranking
        k = min(max(1, top_k), len(self.classes))
        top_predictions = []
        for i in range(k):
            idx = int(top_indices[i])
            cls_name = self.classes[idx]
            prob = float(probs[idx])
            top_predictions.append({
                "rank": i + 1,
                "class": cls_name,
                "code": self.normalizer.get_code(cls_name),
                "display_name": self.normalizer.get_display_name(cls_name),
                "color": self.normalizer.get_color(cls_name),
                "confidence": round(prob, 4),
                "percentage": f"{prob * 100:.2f}%",
            })

        # All class probabilities
        all_probs = {
            cls_name: round(float(p), 4)
            for cls_name, p in zip(self.classes, probs)
        }

        return {
            "is_supported": True,
            "error": None,
            "detected_object": None,
            "prediction": prediction_class,
            "code": self.normalizer.get_code(raw_pred_class) if is_confident else "UNK",
            "display_name": display_prediction,
            "confidence": round(best_prob, 4),
            "percentage": f"{best_prob * 100:.2f}%",
            "status": prediction_status,
            "threshold": thresh,
            "is_confident": is_confident,
            "top_predictions": top_predictions,
            "all_probabilities": all_probs,
        }


def format_cli_output(result: Dict, top_k: int = 3) -> str:
    """Format prediction dictionary into clean, professional terminal report."""
    lines = []
    lines.append("=" * 60)
    if not result.get("is_supported", True):
        lines.append("[ERROR] UNSUPPORTED IMAGE DETECTED")
        lines.append("=" * 60)
        lines.append(f"Status     : {result.get('status', 'Rejected')}")
        lines.append(f"Detected   : {result.get('detected_object', 'Non-construction entity')}")
        lines.append(f"Message    : {result.get('error', 'No construction vehicle found.')}")
        lines.append("-" * 60)
        lines.append("Notice     : This model is strictly calibrated for heavy")
        lines.append("             construction machinery. Supported equipment:")
        lines.append("             [BDZ] Bulldozer       [DTK] Dump Truck")
        lines.append("             [EXC] Excavator       [GRD] Grader")
        lines.append("             [LDR] Loader          [MIX] Mixer Truck")
        lines.append("             [CRN] Mobile Crane    [ROL] Roller")
        lines.append("=" * 60)
        return "\n".join(lines)

    code_tag = f"[{result.get('code', '')}]" if result.get('code') else ""
    lines.append(f"Prediction : {result['display_name']} {code_tag}".strip())
    lines.append(f"Confidence : {result['percentage']}")
    lines.append(f"Status     : {result['status']}")
    lines.append("=" * 60)
    lines.append(f"\nTop {len(result['top_predictions'])} Predictions:")
    for item in result["top_predictions"]:
        lines.append(f"  [{item['rank']}] [{item.get('code', '---')}] {item['display_name']:<14} : {item['percentage']}")
    lines.append("\nAll Class Probabilities:")
    for cls_name, prob in sorted(result["all_probabilities"].items(), key=lambda x: x[1], reverse=True):
        bars = "#" * int(prob * 25)
        lines.append(f"  {cls_name:<14} : {prob * 100:>6.2f}%  |{bars:<25}|")
    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify construction-site vehicle image")
    parser.add_argument("--image", required=True, help="Path to input image file")
    parser.add_argument("--model", default=None, help="Path to model file")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions to display (1, 3, 5)")
    parser.add_argument("--threshold", type=float, default=None, help="Confidence threshold (default 0.50)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of text")
    args = parser.parse_args()

    predictor = VehiclePredictor(model_path=args.model, confidence_threshold=args.threshold)
    res = predictor.predict(args.image, top_k=args.top_k)

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(format_cli_output(res, top_k=args.top_k))
