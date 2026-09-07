import uvicorn
import os
import sys
from pathlib import Path

# Ensure root workspace directory is in python sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

if __name__ == "__main__":
    print("=" * 70)
    print("Synthetic Image Generation & Evaluation Platform Server")
    print("=" * 70)
    print("Web UI Dashboard available at: http://localhost:8000")
    print("API Documentation available at: http://localhost:8000/docs")
    print("=" * 70)
    
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=False)
