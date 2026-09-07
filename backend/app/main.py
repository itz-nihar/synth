from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.app.config import FRONTEND_DIR
from backend.app.api import upload, preprocess, models, evaluate, comparison

app = FastAPI(
    title="Synthetic Image Generation & Evaluation Engine",
    description="End-to-end pipeline for dataset preprocessing, multi-model generation (GAN, Diffusion, VAE-GAN), and automated evaluation (Quality, Utility, Privacy)",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(upload.router)
app.include_router(preprocess.router)
app.include_router(models.router)
app.include_router(evaluate.router)
app.include_router(comparison.router)

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "Synthetic Image Pipeline API"}

# Mount frontend static files
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
