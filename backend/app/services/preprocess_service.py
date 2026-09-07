from pathlib import Path
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from typing import Dict, Any, List
from backend.app.services.dataset_service import get_session_dir, get_session_images, pil_to_base64, SESSION_CACHE

def run_preprocessing_pipeline(
    session_id: str,
    target_size: int = 64,
    train_split_ratio: float = 0.8
) -> Dict[str, Any]:
    raw_images = get_session_images(session_id, "upload")
    if not raw_images:
        raise ValueError(f"No valid raw images found for session {session_id}")
        
    processed_dir = get_session_dir(session_id, "processed") / "real"
    train_dir = get_session_dir(session_id, "processed") / "real_train"
    test_dir = get_session_dir(session_id, "processed") / "real_test"
    
    # Re-create clean folders
    for d in [processed_dir, train_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*"):
            f.unlink()

    preprocessed_images: List[Image.Image] = []
    previews: List[str] = []
    corrupt_count = 0
    
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.Lambda(lambda img: img.convert('RGB'))
    ])
    
    num_train = int(len(raw_images) * train_split_ratio)
    
    for idx, img in enumerate(raw_images):
        try:
            processed_img = transform(img)
            filename = f"img_{idx:04d}.png"
            
            # Save main preprocessed
            processed_img.save(processed_dir / filename)
            
            # Save train vs test split
            if idx < num_train:
                processed_img.save(train_dir / filename)
            else:
                processed_img.save(test_dir / filename)
                
            preprocessed_images.append(processed_img)
            
            if idx < 6:
                previews.append(pil_to_base64(processed_img))
        except Exception:
            corrupt_count += 1
            
    summary = {
        "session_id": session_id,
        "processed_count": len(preprocessed_images),
        "corrupt_count": corrupt_count,
        "target_size": target_size,
        "train_count": num_train,
        "test_count": len(preprocessed_images) - num_train,
        "previews": previews
    }
    
    if session_id in SESSION_CACHE:
        SESSION_CACHE[session_id]["preprocessing"] = summary
    else:
        SESSION_CACHE[session_id] = {"preprocessing": summary}
        
    return summary
