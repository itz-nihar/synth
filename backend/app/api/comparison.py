from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from backend.app.services.comparison_engine import compare_all_models, create_export_package

router = APIRouter(prefix="/api/compare", tags=["Model Comparison & Recommendation"])

class CompareRequest(BaseModel):
    session_id: str
    weight_quality: float = Field(default=0.40)
    weight_utility: float = Field(default=0.40)
    weight_privacy: float = Field(default=0.20)

@router.post("/evaluate_all")
async def evaluate_all_and_compare(req: CompareRequest):
    try:
        res = compare_all_models(
            session_id=req.session_id,
            weight_quality=req.weight_quality,
            weight_utility=req.weight_utility,
            weight_privacy=req.weight_privacy
        )
        return {
            "status": "success",
            "message": f"Successfully evaluated and ranked models. Recommended: {res['recommended_model_name']}",
            "data": res
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/export/{session_id}")
async def export_synthetic_dataset(session_id: str):
    try:
        zip_path = create_export_package(session_id)
        return FileResponse(
            path=zip_path,
            filename=zip_path.name,
            media_type="application/zip"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
