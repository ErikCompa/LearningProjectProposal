from dotenv import load_dotenv

load_dotenv(".env")

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routes import agent_router

# FastAPI app
app = FastAPI()

# Include routers
app.include_router(agent_router)

# Mount static files
# https://fastapi.tiangolo.com/tutorial/static-files/
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
