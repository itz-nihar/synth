import numpy as np
import torch
from PIL import Image
from typing import Dict, Any, List, Optional
from backend.app.utils.metrics_utils import extract_features_from_pil

DATASET_CONTEXT_CACHE: Dict[str, Dict[str, Any]] = {}

def analyze_dataset_context(session_id: str, images: List[Image.Image]) -> Dict[str, Any]:
    if not images:
        raise ValueError(f"No images provided for dataset analysis in session {session_id}")

    total_images = len(images)
    sample_img = images[0]
    width, height = sample_img.size
    channels = len(sample_img.getbands())

    # Extract 512-dim visual embeddings using ResNet vision encoder
    features = extract_features_from_pil(images, batch_size=32) # Shape: (N, 512)

    if len(features) > 0:
        mean_emb = features.mean(axis=0) # Shape: (512,)
        std_emb = features.std(axis=0)   # Shape: (512,)
        
        # Diversity score: average distance of embeddings from dataset mean vector
        dists = np.linalg.norm(features - mean_emb, axis=1)
        diversity_score = float(np.clip(dists.mean() * 10.0, 5.0, 95.0))

        # Unsupervised feature clustering for cluster/category estimation
        clusters_count = 1
        if len(features) >= 4:
            # Simple threshold cosine distance clustering
            norm_feats = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            cos_sim_matrix = np.dot(norm_feats, norm_feats.T)
            mean_sim = float(cos_sim_matrix.mean())
            if mean_sim < 0.70:
                clusters_count = 3
            elif mean_sim < 0.85:
                clusters_count = 2

        if clusters_count == 1:
            detected_type = "Single-Category Visual Domain"
        else:
            detected_type = f"Multi-Category Dataset (~{clusters_count} visual clusters detected)"
    else:
        mean_emb = np.zeros(512)
        diversity_score = 50.0
        clusters_count = 1
        detected_type = "Arbitrary Visual Domain"

    color_space = "RGB Color" if channels == 3 else "Monochrome / Single Channel"

    context = {
        "session_id": session_id,
        "total_images": total_images,
        "resolution": f"{width} × {height}",
        "channels": channels,
        "color_space": color_space,
        "detected_type": detected_type,
        "clusters_count": clusters_count,
        "diversity_score": round(diversity_score, 1),
        "mean_embedding": mean_emb.tolist(),
        "context_status": "Active (Vision Encoder Embedded)"
    }

    DATASET_CONTEXT_CACHE[session_id] = context
    return context

def get_dataset_context(session_id: str) -> Optional[Dict[str, Any]]:
    return DATASET_CONTEXT_CACHE.get(session_id)
