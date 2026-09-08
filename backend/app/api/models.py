from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from backend.app.services.train_service import start_training_job, get_training_status, generate_new_samples_job

router = APIRouter(prefix="/api/models", tags=["Generative Models"])

class ModelTrainRequest(BaseModel):
    session_id: str
    model_type: str = Field(..., description="Generative model choice: 'gan', 'diffusion', or 'vae_gan'")
    epochs: int = Field(default=5, ge=1, le=100)
    num_synthetic_samples: int = Field(default=20, ge=4, le=200)
    image_size: int = Field(default=64)

class ModelGenerateRequest(BaseModel):
    session_id: str
    model_type: str = Field(..., description="Generative model choice: 'gan', 'diffusion', or 'vae_gan'")
    num_synthetic_samples: int = Field(default=20, ge=4, le=200)
    image_size: int = Field(default=128)

@router.post("/train")
async def train_model(req: ModelTrainRequest):
    if req.model_type not in ["gan", "diffusion", "vae_gan"]:
        raise HTTPException(status_code=400, detail="Invalid model_type. Choose 'gan', 'diffusion', or 'vae_gan'")
        
    try:
        job = start_training_job(
            session_id=req.session_id,
            model_type=req.model_type,
            epochs=req.epochs,
            num_synthetic_samples=req.num_synthetic_samples,
            image_size=req.image_size
        )
        return {
            "status": "success",
            "message": f"Started training job for model '{req.model_type}'.",
            "data": job
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/generate")
async def generate_samples(req: ModelGenerateRequest):
    if req.model_type not in ["gan", "diffusion", "vae_gan"]:
        raise HTTPException(status_code=400, detail="Invalid model_type. Choose 'gan', 'diffusion', or 'vae_gan'")

    try:
        res = generate_new_samples_job(
            session_id=req.session_id,
            model_type=req.model_type,
            num_synthetic_samples=req.num_synthetic_samples,
            image_size=req.image_size
        )
        return {
            "status": "success",
            "message": f"Dynamically generated {res['generated_count']} new synthetic samples for '{req.model_type}'.",
            "data": res
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/status/{session_id}/{model_type}")
async def get_model_status(session_id: str, model_type: str):
    status = get_training_status(session_id, model_type)
    return {"status": "success", "data": status}
