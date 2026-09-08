import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
from backend.app.services.dataset_service import get_session_images, SESSION_CACHE
from backend.app.utils.metrics_utils import (
    extract_features_from_pil,
    compute_fid,
    compute_ssim_psnr_metrics,
    compute_histogram_similarity,
    compute_tstr_utility,
    compute_nndr_privacy
)

EVALUATION_CACHE: Dict[str, Dict[str, Any]] = {}

def compute_semantic_similarity(real_feats: np.ndarray, syn_feats: np.ndarray) -> float:
    if len(real_feats) == 0 or len(syn_feats) == 0:
        return 0.0
    r_mean = real_feats.mean(axis=0)
    s_mean = syn_feats.mean(axis=0)
    
    r_norm = np.linalg.norm(r_mean) + 1e-8
    s_norm = np.linalg.norm(s_mean) + 1e-8
    
    cosine_sim = float(np.dot(r_mean, s_mean) / (r_norm * s_norm))
    return float(np.clip(cosine_sim, 0.0, 1.0))

def validate_synthetic_quality(synthetic_imgs: List[Image.Image], syn_feats: np.ndarray, fid: float, sem_sim: float = 1.0) -> Tuple[bool, str]:
    """
    Quality Validation Guard: Checks whether synthetic dataset is valid structure matching dataset context or garbage/noise.
    """
    if len(synthetic_imgs) == 0 or len(syn_feats) == 0:
        return False, "Synthetic image dataset is empty."
        
    feat_std = float(syn_feats.std(axis=0).mean()) if len(syn_feats) > 1 else 0.0
    
    if fid > 350.0:
        return False, f"FID metric is extremely high ({round(fid, 1)} > 350.0), indicating un-converged noise."
        
    if sem_sim < 0.40:
        return False, f"Semantic Similarity is too low ({round(sem_sim, 2)} < 0.40), generated output does not match dataset context."

    if feat_std < 0.001:
        return False, "Synthetic dataset exhibits feature collapse (mode collapse)."
        
    return True, "Quality validation passed."

def evaluate_model_synthetic_dataset(session_id: str, model_type: str) -> Dict[str, Any]:
    real_train_imgs = get_session_images(session_id, "processed", "real_train")
    real_test_imgs = get_session_images(session_id, "processed", "real_test")
    real_all_imgs = get_session_images(session_id, "processed", "real")
    
    synthetic_imgs = get_session_images(session_id, "generated", model_type)
    
    if not real_all_imgs:
        raise ValueError(f"No preprocessed real images found for session {session_id}.")
    if not synthetic_imgs:
        raise ValueError(f"No synthetic images found for model {model_type} in session {session_id}.")

    # 1. Feature Extraction
    real_feats = extract_features_from_pil(real_all_imgs)
    syn_feats = extract_features_from_pil(synthetic_imgs)
    
    # 2. Semantic Similarity & Quality Metrics
    sem_sim = compute_semantic_similarity(real_feats, syn_feats)
    fid = compute_fid(real_feats, syn_feats)
    ssim, psnr = compute_ssim_psnr_metrics(real_all_imgs, synthetic_imgs)
    hist_sim = compute_histogram_similarity(real_all_imgs, synthetic_imgs)
    
    # Run Quality Validation Guard
    is_valid, validation_msg = validate_synthetic_quality(synthetic_imgs, syn_feats, fid, sem_sim)
    
    if not is_valid:
        result = {
            "session_id": session_id,
            "model_type": model_type,
            "validation_status": "FAIL",
            "validation_message": validation_msg,
            "semantic_similarity": round(sem_sim, 4),
            "quality": {"score": 0.0, "fid": round(fid, 2), "ssim": round(ssim, 4), "psnr": round(psnr, 2), "histogram_similarity": round(hist_sim, 4)},
            "utility": {"score": 0.0, "tstr_accuracy": 0.0, "trtr_accuracy": 0.0, "utility_ratio": 0.0},
            "privacy": {"score": 0.0, "nndr_score": 0.0, "copy_leakage_percent": 0.0}
        }
    else:
        # Quality Score: higher semantic similarity & lower FID & higher SSIM = higher score
        fid_score_component = max(0.0, 100.0 - (fid * 1.5))
        ssim_component = max(0.0, ssim * 100.0)
        sem_component = max(0.0, sem_sim * 100.0)
        quality_score = float(np.clip(0.4 * sem_component + 0.3 * fid_score_component + 0.3 * ssim_component, 0.0, 100.0))
        
        # 3. Utility Metrics (TSTR)
        utility_metrics = compute_tstr_utility(real_train_imgs or real_all_imgs, synthetic_imgs)
        u_ratio = utility_metrics.get("utility_ratio", 0.0)
        utility_score = float(np.clip(u_ratio * 100.0, 0.0, 100.0))
        
        # 4. Privacy Metrics (NNDR & Leakage)
        privacy_metrics = compute_nndr_privacy(real_feats, syn_feats)
        privacy_score = privacy_metrics.get("privacy_score", 50.0)
        
        result = {
            "session_id": session_id,
            "model_type": model_type,
            "validation_status": "PASS",
            "validation_message": validation_msg,
            "semantic_similarity": round(sem_sim, 4),
            "quality": {
                "score": round(quality_score, 2),
                "fid": round(fid, 2),
                "ssim": round(ssim, 4),
                "psnr": round(psnr, 2),
                "histogram_similarity": round(hist_sim, 4)
            },
            "utility": {
                "score": round(utility_score, 2),
                "tstr_accuracy": utility_metrics.get("tstr_accuracy", 0.0),
                "trtr_accuracy": utility_metrics.get("trtr_accuracy", 0.0),
                "utility_ratio": u_ratio
            },
            "privacy": {
                "score": round(privacy_score, 2),
                "nndr_score": privacy_metrics.get("nndr_score", 0.0),
                "copy_leakage_percent": privacy_metrics.get("copy_leakage_percent", 0.0)
            }
        }
    
    cache_key = f"{session_id}_{model_type}"
    EVALUATION_CACHE[cache_key] = result
    return result

def get_evaluation_results(session_id: str, model_type: str) -> Optional[Dict[str, Any]]:
    cache_key = f"{session_id}_{model_type}"
    return EVALUATION_CACHE.get(cache_key)
