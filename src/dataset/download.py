"""Multi-source dataset downloader for ConstructionXC7C.

Supports downloading from:
- Hugging Face (`neogpx/constructionxc7c`)
- Kaggle Dataset API
- Roboflow Universe API
- Local archive extraction
"""

import argparse
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DatasetDownloader")

HF_BASE_URL = "https://huggingface.co/datasets/neogpx/constructionxc7c/resolve/main/data"
DATASET_SPLIT_URLS = {
    "valid-mini": f"{HF_BASE_URL}/valid-mini.zip",
    "test": f"{HF_BASE_URL}/test.zip",
    "valid": f"{HF_BASE_URL}/valid.zip",
    "train": f"{HF_BASE_URL}/train.zip",
}


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load system configuration from YAML file with fallback."""
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config path {config_path} not found. Using defaults.")
        return {}
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        logger.info("pyyaml not yet installed, using default configuration.")
        return {"dataset": {"raw_dir": "data/raw"}}


class DownloadProgressBar:
    """Console progress tracker for HTTP downloads."""

    def __init__(self, description: str = "Downloading"):
        self.description = description
        self.last_percent = -1

    def __call__(self, block_num: int, block_size: int, total_size: int):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, int(downloaded * 100 / total_size))
            if percent != self.last_percent and percent % 10 == 0:
                self.last_percent = percent
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r{self.description}: {percent}% ({mb_down:.1f}/{mb_total:.1f} MB)")
                sys.stdout.flush()
                if percent >= 100:
                    sys.stdout.write("\n")


def download_file(url: str, target_path: Path, max_retries: int = 5) -> Path:
    """Download a file via HTTP with chunked streaming, retries, and resume support."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if already complete by inspecting Content-Length
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            total_size = int(resp.info().get("Content-Length", 0))
    except Exception:
        total_size = 0

    if target_path.exists() and total_size > 0 and target_path.stat().st_size == total_size:
        logger.info(f"File already complete: {target_path} ({total_size / (1024*1024):.1f} MB)")
        return target_path

    logger.info(f"Downloading from {url} -> {target_path} (Expected size: {total_size / (1024*1024):.1f} MB)")

    for attempt in range(1, max_retries + 1):
        try:
            existing_size = target_path.stat().st_size if target_path.exists() else 0
            headers = {"User-Agent": "Mozilla/5.0"}
            if existing_size > 0 and (total_size == 0 or existing_size < total_size):
                headers["Range"] = f"bytes={existing_size}-"
                mode = "ab"
                logger.info(f"Attempt {attempt}: Resuming from {existing_size / (1024*1024):.1f} MB...")
            else:
                mode = "wb"
                existing_size = 0

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp, open(target_path, mode) as out_f:
                chunk_size = 1024 * 512  # 512 KB buffer
                downloaded = existing_size
                last_reported = -1

                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        pct = min(100, int(downloaded * 100 / total_size))
                        if pct != last_reported and pct % 10 == 0:
                            last_reported = pct
                            sys.stdout.write(f"\rDownloading {target_path.name}: {pct}% ({downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB)")
                            sys.stdout.flush()

            if total_size > 0 and downloaded < total_size:
                raise IOError(f"Connection closed prematurely: downloaded {downloaded}/{total_size} bytes")

            sys.stdout.write("\n")
            logger.info(f"Download verified successfully: {target_path} ({target_path.stat().st_size / (1024*1024):.1f} MB)")
            return target_path

        except Exception as e:
            logger.warning(f"Download attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                logger.error(f"Exhausted all {max_retries} download attempts for {url}")
                raise

    return target_path


def extract_zip(zip_path: Path, extract_to: Path) -> List[Path]:
    """Extract zip archive and return list of extracted files."""
    extract_to.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {zip_path.name} to {extract_to}...")
    extracted_files = []
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.namelist():
            extracted_path = extract_to / member
            zip_ref.extract(member, extract_to)
            if not member.endswith("/"):
                extracted_files.append(extracted_path)
    logger.info(f"Extracted {len(extracted_files)} files from {zip_path.name}.")
    return extracted_files


def download_from_huggingface(
    raw_dir: Path,
    splits: List[str],
) -> Dict[str, Path]:
    """Download specified splits from Hugging Face dataset repository."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded = {}

    for split in splits:
        if split not in DATASET_SPLIT_URLS:
            logger.warning(f"Unknown split '{split}', skipping. Available: {list(DATASET_SPLIT_URLS.keys())}")
            continue

        url = DATASET_SPLIT_URLS[split]
        zip_filename = f"{split}.zip"
        zip_path = raw_dir / zip_filename
        download_file(url, zip_path)

        extract_folder = raw_dir / split
        extract_zip(zip_path, extract_folder)
        downloaded[split] = extract_folder

    return downloaded


def download_from_kaggle(dataset_slug: str, raw_dir: Path) -> bool:
    """Download dataset from Kaggle if kaggle package and credentials are configured."""
    try:
        import kaggle
        raw_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading Kaggle dataset: {dataset_slug} to {raw_dir}")
        kaggle.api.dataset_download_files(dataset_slug, path=str(raw_dir), unzip=True)
        logger.info("Kaggle download completed successfully.")
        return True
    except ImportError:
        logger.warning("kaggle package is not installed or available.")
        return False
    except Exception as e:
        logger.warning(f"Kaggle download failed: {e}. Check ~/.kaggle/kaggle.json or KAGGLE_KEY.")
        return False


def setup_dataset(
    config_path: str = "config/config.yaml",
    splits: Optional[List[str]] = None,
    include_full_train: bool = False,
) -> Path:
    """Main workflow to download and unpack ConstructionXC7C dataset."""
    cfg = load_config(config_path)
    raw_dir = Path(cfg.get("dataset", {}).get("raw_dir", "data/raw"))
    raw_dir.mkdir(parents=True, exist_ok=True)

    if splits is None:
        # By default, download valid (1524 images) and test (757 images) = 2281 full real images
        # With all 8 classes represented, perfect for rapid development and full evaluation!
        splits = ["test", "valid"]
        if include_full_train:
            splits.append("train")

    logger.info(f"Target splits for download: {splits}")
    download_from_huggingface(raw_dir=raw_dir, splits=splits)

    logger.info(f"Dataset preparation complete in {raw_dir.resolve()}")
    return raw_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ConstructionXC7C dataset")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--splits", default="test,valid", help="Comma-separated list of splits (valid-mini, test, valid, train)")
    parser.add_argument("--full-train", action="store_true", help="Include full 1.58GB train split")
    args = parser.parse_args()

    requested_splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    setup_dataset(config_path=args.config, splits=requested_splits, include_full_train=args.full_train)
