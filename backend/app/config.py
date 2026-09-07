import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
STORAGE_DIR = BASE_DIR / "storage"

UPLOAD_DIR = STORAGE_DIR / "uploads"
PROCESSED_DIR = STORAGE_DIR / "processed"
GENERATED_DIR = STORAGE_DIR / "generated"
CHECKPOINT_DIR = STORAGE_DIR / "checkpoints"
EXPORTS_DIR = STORAGE_DIR / "exports"

# Create directories if they don't exist
for folder in [STORAGE_DIR, UPLOAD_DIR, PROCESSED_DIR, GENERATED_DIR, CHECKPOINT_DIR, EXPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_IMAGE_SIZE = 64
