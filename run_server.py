"""
Simple FastAPI server for Lip Sync API
Run with: python run_server.py
"""

import sys
import os
from pathlib import Path

# Set environment variables for testing
os.environ["PROJECT_NAME"] = "LipSync API"
os.environ["POSTGRES_SERVER"] = "localhost"
os.environ["POSTGRES_USER"] = "postgres"
os.environ["POSTGRES_PASSWORD"] = "changethis"
os.environ["POSTGRES_DB"] = "app"
os.environ["FIRST_SUPERUSER"] = "admin@example.com"
os.environ["FIRST_SUPERUSER_PASSWORD"] = "changethis"
os.environ["ENVIRONMENT"] = "local"

# Add paths
backend_path = Path(__file__).parent / "backend"
sadtalker_path = Path(__file__).parent / "models" / "sadtalker"
sys.path.insert(0, str(backend_path))
sys.path.insert(0, str(sadtalker_path))

# Change to backend directory
os.chdir(str(backend_path))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the lipsync router directly
from app.api.routes.lipsync import router as lipsync_router

# Create a simple app without database dependencies
app = FastAPI(
    title="LipSync API",
    description="API for generating lip-synced videos using Wav2Lip and SadTalker",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include only the lipsync router
app.include_router(lipsync_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "LipSync API Server",
        "docs": "/docs",
        "endpoints": {
            "generate": "POST /api/v1/lipsync/generate",
            "status": "GET /api/v1/lipsync/status/{job_id}",
            "models": "GET /api/v1/lipsync/models",
            "download": "GET /api/v1/lipsync/download/{job_id}"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("=" * 50)
    print("Starting LipSync API Server")
    print("=" * 50)
    print(f"API Documentation: http://localhost:8000/docs")
    print(f"Generate endpoint: POST http://localhost:8000/api/v1/lipsync/generate")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
