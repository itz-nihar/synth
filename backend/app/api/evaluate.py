from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from backend.app.services.evaluation_engine import evaluate_model_synthetic_dataset, get_evaluation_results

router = APIRouter(prefix="/api/evaluate", tags=["Evaluation Engine"])

class EvaluateRequest(BaseModel):
    session_id: str
    model_type: str = Field(..., description="Model type: 'gan', 'diffusion', or 'vae_gan'")

@router.post("/run")
async def run_evaluation(req: EvaluateRequest):
    try:
        res = evaluate_model_synthetic_dataset(req.session_id, req.model_type)
        return {
            "status": "success",
            "message": f"Successfully evaluated synthetic dataset for model {req.model_type}.",
            "data": res
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/results/{session_id}/{model_type}")
async def get_results(session_id: str, model_type: str):
    res = get_evaluation_results(session_id, model_type)
    if res:
        return {"status": "success", "data": res}
    raise HTTPException(status_code=404, detail="Evaluation results not found for model and session")
