import json
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional
from backend.app.config import UPLOAD_DIR, PROCESSED_DIR, GENERATED_DIR, CHECKPOINT_DIR
from backend.app.utils.file_utils import generate_session_id, extract_zip_dataset, is_valid_image

# Memory store for active session states
SESSION_CACHE: Dict[str, Dict[str, Any]] = {}

def get_session_dir(session_id: str, category: str = "upload") -> Path:
    if category == "upload":
        p = UPLOAD_DIR / session_id
    elif category == "processed":
        p = PROCESSED_DIR / session_id
    elif category == "generated":
        p = GENERATED_DIR / session_id
    elif category == "checkpoint":
        p = CHECKPOINT_DIR / session_id
    else:
        p = UPLOAD_DIR / session_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    buffered = BytesIO()
    img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/{format.lower()};base64,{img_str}"

def process_uploaded_files(session_id: Optional[str], file_path: Path, is_zip: bool = True) -> Dict[str, Any]:
    if not session_id:
        session_id = generate_session_id()
        
    upload_dir = get_session_dir(session_id, "upload")
    
    if is_zip:
        image_paths = extract_zip_dataset(file_path, upload_dir)
    else:
        image_paths = [file_path] if is_valid_image(file_path) else []
        
    # Analyze raw image stats
    shapes = []
    previews = []
    
    for idx, p in enumerate(image_paths):
        try:
            with Image.open(p) as img:
                shapes.append((img.width, img.height, len(img.getbands())))
                if idx < 6: # Convert first 6 images to base64 preview
                    previews.append(pil_to_base64(img.resize((128, 128))))
        except Exception:
            continue
            
    stats = {
        "session_id": session_id,
        "total_images": len(image_paths),
        "shapes": shapes,
        "previews": previews,
        "image_paths": [str(p) for p in image_paths]
    }
    
    SESSION_CACHE[session_id] = stats
    return stats

def get_session_images(session_id: str, category: str = "upload", subfolder: str = "") -> List[Image.Image]:
    target_dir = get_session_dir(session_id, category)
    if subfolder:
        target_dir = target_dir / subfolder
        
    if not target_dir.exists():
        return []
        
    images = []
    for file in sorted(target_dir.glob("*")):
        if is_valid_image(file):
            try:
                img = Image.open(file).copy()
                images.append(img)
            except Exception:
                continue
    return images
