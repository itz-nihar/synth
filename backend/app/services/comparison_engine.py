import json
from pathlib import Path
from typing import Dict, Any, List
from backend.app.services.evaluation_engine import evaluate_model_synthetic_dataset, get_evaluation_results
from backend.app.services.dataset_service import get_session_dir, get_session_images, pil_to_base64
from backend.app.utils.file_utils import create_zip_archive

COMPARISON_CACHE: Dict[str, Dict[str, Any]] = {}

MODEL_NAMES = {
    "gan": "GAN (DCGAN/WGAN)",
    "diffusion": "Diffusion Model (DDPM)",
    "vae_gan": "VAE/GAN Hybrid"
}

def compare_all_models(
    session_id: str,
    weight_quality: float = 0.40,
    weight_utility: float = 0.40,
    weight_privacy: float = 0.20
) -> Dict[str, Any]:
    models_to_eval = ["gan", "diffusion", "vae_gan"]
    evaluations: List[Dict[str, Any]] = []

    for m in models_to_eval:
        eval_res = get_evaluation_results(session_id, m)
        if not eval_res:
            try:
                eval_res = evaluate_model_synthetic_dataset(session_id, m)
            except Exception:
                # If model hasn't been generated yet, skip or mock baseline
                continue
        
        q_score = eval_res["quality"]["score"]
        u_score = eval_res["utility"]["score"]
        p_score = eval_res["privacy"]["score"]

        composite = (weight_quality * q_score) + (weight_utility * u_score) + (weight_privacy * p_score)
        
        eval_res["composite_score"] = round(composite, 2)
        eval_res["model_name"] = MODEL_NAMES.get(m, m)
        
        # Add preview grid
        gen_imgs = get_session_images(session_id, "generated", m)
        eval_res["previews"] = [pil_to_base64(img) for img in gen_imgs[:4]]
        
        evaluations.append(eval_res)

    if not evaluations:
        raise ValueError(f"No models have been trained or generated yet for session {session_id}.")

    # Sort models by composite score descending
    evaluations.sort(key=lambda x: x["composite_score"], reverse=True)
    
    # Assign ranks
    for rank, item in enumerate(evaluations, 1):
        item["rank"] = rank

    recommended_model = evaluations[0]
    
    # Generate Rationale Explanation
    rec_type = recommended_model["model_type"]
    rec_name = recommended_model["model_name"]
    rationale = (
        f"{rec_name} is the RECOMMENDED MODEL with the highest overall Composite Score of {recommended_model['composite_score']}/100. "
        f"It achieved an Image Quality Score of {recommended_model['quality']['score']}/100 (FID: {recommended_model['quality']['fid']}), "
        f"a Utility Score of {recommended_model['utility']['score']}/100 (TSTR Accuracy: {recommended_model['utility']['tstr_accuracy']}), "
        f"and a Privacy Score of {recommended_model['privacy']['score']}/100."
    )

    comparison_report = {
        "session_id": session_id,
        "recommended_model_type": rec_type,
        "recommended_model_name": rec_name,
        "rationale": rationale,
        "weights": {
            "quality": weight_quality,
            "utility": weight_utility,
            "privacy": weight_privacy
        },
        "rankings": evaluations
    }

    COMPARISON_CACHE[session_id] = comparison_report
    return comparison_report

def create_export_package(session_id: str) -> Path:
    report = COMPARISON_CACHE.get(session_id)
    if not report:
        report = compare_all_models(session_id)
        
    rec_type = report["recommended_model_type"]
    
    # Build temporary export directory
    export_bundle_dir = get_session_dir(session_id, "exports") / f"bundle_{rec_type}"
    export_bundle_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy recommended synthetic images
    syn_imgs = get_session_images(session_id, "generated", rec_type)
    syn_out_dir = export_bundle_dir / "synthetic_dataset"
    syn_out_dir.mkdir(parents=True, exist_ok=True)
    for idx, img in enumerate(syn_imgs):
        img.save(syn_out_dir / f"synthetic_{idx:04d}.png")
        
    # 2. Write evaluation report JSON
    with open(export_bundle_dir / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    # 3. Create output ZIP archive
    zip_out = get_session_dir(session_id, "exports") / f"synthetic_dataset_{session_id}_{rec_type}.zip"
    return create_zip_archive(export_bundle_dir, zip_out)
