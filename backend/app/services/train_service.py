import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
import threading
import time

from backend.app.services.dataset_service import get_session_dir, get_session_images, pil_to_base64
from backend.app.models.gan import GANModel
from backend.app.models.diffusion import DiffusionModel
from backend.app.models.vae_gan import VAEGANModel
from backend.app.config import DEVICE

# In-memory training status store
TRAINING_STATUS: Dict[str, Dict[str, Any]] = {}

def get_status_key(session_id: str, model_type: str) -> str:
    return f"{session_id}_{model_type}"

def prepare_pytorch_dataloader(session_id: str, image_size: int = 64, batch_size: int = 16) -> Tuple[DataLoader, bool]:
    processed_imgs = get_session_images(session_id, "processed", "real")
    if not processed_imgs:
        raise ValueError(f"No preprocessed images found for session {session_id}. Please run preprocessing first.")
        
    # Check if dataset is monochrome (e.g. Brain Tumor MRI images)
    sample_arr = np.array(processed_imgs[0].convert('RGB'))
    channel_diff = np.abs(sample_arr[:, :, 0] - sample_arr[:, :, 1]).max()
    is_monochrome = bool(channel_diff < 5) # < 5 pixel tolerance
        
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Map [0,1] -> [-1,1]
    ])
    
    tensors = [transform(img.convert('RGB')) for img in processed_imgs]
    tensor_dataset = TensorDataset(torch.stack(tensors))
    loader = DataLoader(tensor_dataset, batch_size=min(batch_size, len(tensors)), shuffle=True)
    return loader, is_monochrome

def train_and_generate_task(
    session_id: str,
    model_type: str, # "gan", "diffusion", "vae_gan"
    epochs: int = 5,
    num_synthetic_samples: int = 20,
    image_size: int = 64
):
    key = get_status_key(session_id, model_type)
    TRAINING_STATUS[key] = {
        "session_id": session_id,
        "model_type": model_type,
        "status": "training",
        "current_epoch": 0,
        "total_epochs": epochs,
        "loss_g": 0.0,
        "loss_d": 0.0,
        "progress_percent": 0,
        "previews": [],
        "generated_count": 0,
        "error": None
    }
    
    try:
        dataloader, is_monochrome = prepare_pytorch_dataloader(session_id, image_size=image_size)
        
        # Instantiate model
        if model_type == "gan":
            model = GANModel(image_size=image_size, is_monochrome=is_monochrome)
        elif model_type == "diffusion":
            model = DiffusionModel(image_size=image_size, is_monochrome=is_monochrome)
        elif model_type == "vae_gan":
            model = VAEGANModel(image_size=image_size, is_monochrome=is_monochrome)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        def progress_callback(ep, total_ep, lg, ld):
            pct = int((ep / total_ep) * 80) # Training is 80% of progress
            TRAINING_STATUS[key].update({
                "current_epoch": ep,
                "loss_g": round(lg, 4),
                "loss_d": round(ld, 4),
                "progress_percent": pct
            })

        # Step 1: Train Model
        train_res = model.train_model(dataloader, epochs=epochs, progress_callback=progress_callback)
        
        # Step 2: Save Checkpoint
        ckpt_path = get_session_dir(session_id, "checkpoint") / f"{model_type}.pt"
        model.save_checkpoint(ckpt_path)
        
        # Step 3: Generate Synthetic Image Samples
        TRAINING_STATUS[key]["status"] = "generating"
        TRAINING_STATUS[key]["progress_percent"] = 85
        
        synthetic_imgs = model.generate_samples(num_samples=num_synthetic_samples)
        
        # Save synthetic images to generated folder
        gen_dir = get_session_dir(session_id, "generated") / model_type
        gen_dir.mkdir(parents=True, exist_ok=True)
        # Clear existing
        for f in gen_dir.glob("*"):
            f.unlink()
            
        previews = []
        for idx, img in enumerate(synthetic_imgs):
            filename = f"syn_{idx:04d}.png"
            img.save(gen_dir / filename)
            if idx < 6:
                previews.append(pil_to_base64(img))
                
        TRAINING_STATUS[key].update({
            "status": "completed",
            "progress_percent": 100,
            "loss_g": train_res.get("loss_g", 0.0),
            "loss_d": train_res.get("loss_d", 0.0),
            "previews": previews,
            "generated_count": len(synthetic_imgs)
        })
        
    except Exception as e:
        TRAINING_STATUS[key].update({
            "status": "failed",
            "error": str(e),
            "progress_percent": 0
        })

def start_training_job(
    session_id: str,
    model_type: str,
    epochs: int = 5,
    num_synthetic_samples: int = 20,
    image_size: int = 64
) -> Dict[str, Any]:
    key = get_status_key(session_id, model_type)
    
    # Launch in background thread
    t = threading.Thread(
        target=train_and_generate_task,
        args=(session_id, model_type, epochs, num_synthetic_samples, image_size),
        daemon=True
    )
    t.start()
    
    return {
        "session_id": session_id,
        "model_type": model_type,
        "status": "started"
    }

def get_training_status(session_id: str, model_type: str) -> Dict[str, Any]:
    key = get_status_key(session_id, model_type)
    if key in TRAINING_STATUS:
        return TRAINING_STATUS[key]
    return {
        "session_id": session_id,
        "model_type": model_type,
        "status": "not_started",
        "progress_percent": 0
    }
