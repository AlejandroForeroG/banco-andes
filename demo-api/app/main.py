from contextlib import asynccontextmanager
from fastapi import FastAPI
from app import database
from app.routers import finanzas

@asynccontextmanager
async def lifespan(app: FastAPI):
    database.load_seed()
    yield

app = FastAPI(title="Banco Andes — API de demostración", version="1.0.0", lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok"}

app.include_router(finanzas.router, prefix="/finanzas", tags=["finanzas"])
