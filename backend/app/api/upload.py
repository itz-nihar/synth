import os
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from backend.app.services.dataset_service import process_uploaded_files, generate_session_id, get_session_dir, SESSION_CACHE

router = APIRouter(prefix="/api/dataset", tags=["Upload & Dataset"])

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    if not session_id:
        session_id = generate_session_id()

    upload_dir = get_session_dir(session_id, "upload")
    temp_file_path = upload_dir / file.filename

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    is_zip = file.filename.lower().endswith(".zip")
    
    try:
        stats = process_uploaded_files(session_id, temp_file_path, is_zip=is_zip)
        return {
            "status": "success",
            "message": f"Successfully uploaded and extracted {stats['total_images']} images.",
            "data": stats
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/manifest/{session_id}")
async def get_manifest(session_id: str):
    if session_id in SESSION_CACHE:
        return {"status": "success", "data": SESSION_CACHE[session_id]}
    
    # Try reading files from directory if not in memory cache
    upload_dir = get_session_dir(session_id, "upload")
    if not upload_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
        
    imgs = list(upload_dir.glob("*"))
    return {
        "status": "success",
        "data": {
            "session_id": session_id,
            "total_images": len(imgs),
            "image_paths": [str(p) for p in imgs]
        }
    }
