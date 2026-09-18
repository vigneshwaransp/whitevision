"""Class normalization layer mapping dataset categories to canonical classes."""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import yaml

logger = logging.getLogger("ClassNormalization")


class ClassNormalizer:
    """Normalizes raw class labels from datasets into canonical project vehicle classes."""

    def __init__(self, config_path: str = "config/classes.yaml"):
        self.config_path = Path(config_path)
        self.canonical_classes: List[str] = []
        self.alias_to_canonical: Dict[str, str] = {}
        self.class_metadata: Dict[str, dict] = {}
        self.canonical_to_id: Dict[str, int] = {}
        self.id_to_canonical: Dict[int, str] = {}
        self._load_config()

    def _load_config(self):
        if not self.config_path.exists():
            # Fallback default definitions
            default_classes = [
                "bulldozer",
                "dump_truck",
                "excavator",
                "grader",
                "loader",
                "mixer_truck",
                "mobile_crane",
                "roller",
            ]
            self.canonical_classes = default_classes
            for idx, c in enumerate(default_classes):
                self.canonical_to_id[c] = idx
                self.id_to_canonical[idx] = c
                self.alias_to_canonical[c] = c
                self.alias_to_canonical[c.replace("_", " ")] = c
                self.alias_to_canonical[c.replace("_", "-")] = c
            return

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        classes_list = data.get("classes", [])
        for item in classes_list:
            canonical = item["canonical"]
            cls_id = int(item["id"])
            self.canonical_classes.append(canonical)
            self.canonical_to_id[canonical] = cls_id
            self.id_to_canonical[cls_id] = canonical
            self.class_metadata[canonical] = item

            # Register canonical itself
            self._register_alias(canonical, canonical)

            # Register configured aliases
            for alias in item.get("aliases", []):
                self._register_alias(alias, canonical)

    def _clean_str(self, s: str) -> str:
        return s.strip().lower().replace("-", "_").replace(" ", "_")

    def _register_alias(self, raw_alias: str, canonical: str):
        cleaned = self._clean_str(raw_alias)
        self.alias_to_canonical[cleaned] = canonical
        # Also store original lower
        self.alias_to_canonical[raw_alias.strip().lower()] = canonical

    def normalize(self, raw_label: str) -> Optional[str]:
        """Convert any vehicle category string to its canonical class name."""
        if not raw_label:
            return None
        cleaned = self._clean_str(raw_label)
        if cleaned in self.alias_to_canonical:
            return self.alias_to_canonical[cleaned]

        lower = raw_label.strip().lower()
        if lower in self.alias_to_canonical:
            return self.alias_to_canonical[lower]

        # Partial matching heuristic
        for alias, canonical in self.alias_to_canonical.items():
            if alias in cleaned or cleaned in alias:
                return canonical

        return None

    def get_classes(self) -> List[str]:
        """Return the sorted list of canonical class names."""
        return sorted(self.canonical_classes, key=lambda c: self.canonical_to_id.get(c, 0))

    def get_display_name(self, canonical: str) -> str:
        """Return human-readable display name for class."""
        meta = self.class_metadata.get(canonical, {})
        return meta.get("display_name", canonical.replace("_", " ").title())

    def get_code(self, canonical: str) -> str:
        """Return 3-letter uppercase identifier code for class."""
        meta = self.class_metadata.get(canonical, {})
        return meta.get("code", canonical[:3].upper())

    def get_emoji(self, canonical: str) -> str:
        """Return empty string (emojis disabled)."""
        return ""

    def get_color(self, canonical: str) -> str:
        """Return hex color code for class."""
        meta = self.class_metadata.get(canonical, {})
        return meta.get("color", "#3498DB")


# Global default instance
_default_normalizer: Optional[ClassNormalizer] = None


def get_normalizer(config_path: str = "config/classes.yaml") -> ClassNormalizer:
    global _default_normalizer
    if _default_normalizer is None or str(_default_normalizer.config_path) != config_path:
        _default_normalizer = ClassNormalizer(config_path)
    return _default_normalizer


def normalize_class_name(raw_label: str, config_path: str = "config/classes.yaml") -> Optional[str]:
    return get_normalizer(config_path).normalize(raw_label)
