import os
import zipfile
import shutil
import uuid
from pathlib import Path
from PIL import Image, ImageOps
from backend.app.config import ALLOWED_EXTENSIONS

def generate_session_id() -> str:
    return str(uuid.uuid4())[:8]

def is_valid_image(file_path: Path) -> bool:
    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def extract_zip_dataset(zip_path: Path, target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted_images = []
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            # Security check for zip slip
            filename = os.path.basename(member)
            if not filename or filename.startswith('.'):
                continue
            
            ext = Path(filename).suffix.lower()
            if ext in ALLOWED_EXTENSIONS:
                extracted_path = target_dir / filename
                # If filename collision, append unique suffix
                if extracted_path.exists():
                    extracted_path = target_dir / f"{Path(filename).stem}_{uuid.uuid4().hex[:4]}{ext}"
                
                with zip_ref.open(member) as source, open(extracted_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                
                if is_valid_image(extracted_path):
                    extracted_images.append(extracted_path)
                else:
                    if extracted_path.exists():
                        extracted_path.unlink()
    return extracted_images

def save_image_grid(images: list[Image.Image], save_path: Path, cols: int = 4):
    if not images:
        return
    rows = (len(images) + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new('RGB', (cols * w, rows * h), (255, 255, 255))
    
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        grid.paste(img, (c * w, r * h))
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(save_path)

def create_zip_archive(source_dir: Path, zip_out_path: Path) -> Path:
    zip_out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_out_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_p = Path(root) / file
                arcname = file_p.relative_to(source_dir)
                zipf.write(file_p, arcname)
    return zip_out_path
