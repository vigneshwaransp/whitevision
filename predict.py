"""Root CLI entrypoint for Construction Vehicle Classifier inference.

Usage:
    python predict.py --image path/to/image.jpg [--top_k 3] [--threshold 0.50] [--json]
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.inference.predict import VehiclePredictor, format_cli_output

def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Construction-Site Vehicle Image Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", "-i", required=True, help="Path to input construction vehicle image")
    parser.add_argument("--model", "-m", default=None, help="Optional path to model .keras file")
    parser.add_argument("--top_k", "-k", type=int, default=3, help="Top-K predictions to display (e.g. 1, 3, 5)")
    parser.add_argument("--threshold", "-t", type=float, default=0.50, help="Minimum confidence threshold (0.0 - 1.0)")
    parser.add_argument("--json", action="store_true", help="Print output in JSON format")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found at '{image_path}'", file=sys.stderr)
        sys.exit(1)

    try:
        predictor = VehiclePredictor(model_path=args.model, confidence_threshold=args.threshold)
        result = predictor.predict(image_path, top_k=args.top_k, threshold=args.threshold)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(format_cli_output(result, top_k=args.top_k))

    except Exception as e:
        print(f"Inference error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
