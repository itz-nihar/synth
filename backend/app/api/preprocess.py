from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from backend.app.services.preprocess_service import run_preprocessing_pipeline, SESSION_CACHE

router = APIRouter(prefix="/api/preprocess", tags=["Preprocessing & Validation"])

class PreprocessRequest(BaseModel):
    session_id: str
    target_size: int = Field(default=64, description="Target pixel width/height (32, 64, 128)")
    train_split_ratio: float = Field(default=0.8, description="Train / validation split ratio")

@router.post("/run")
async def run_preprocess(req: PreprocessRequest):
    try:
        res = run_preprocessing_pipeline(
            session_id=req.session_id,
            target_size=req.target_size,
            train_split_ratio=req.train_split_ratio
        )
        return {
            "status": "success",
            "message": f"Successfully preprocessed {res['processed_count']} images to {req.target_size}x{req.target_size}.",
            "data": res
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/status/{session_id}")
async def get_preprocess_status(session_id: str):
    if session_id in SESSION_CACHE and "preprocessing" in SESSION_CACHE[session_id]:
        return {"status": "success", "data": SESSION_CACHE[session_id]["preprocessing"]}
    raise HTTPException(status_code=404, detail="Preprocessing status not found for session")
