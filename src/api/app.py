from fastapi import FastAPI

from src.api.routes.train import router as train_router

app = FastAPI(
    title="ML Training API",
    description="API for training LSTM models",
    version="1.0.0",
)

app.include_router(train_router, prefix="/train", tags=["Training"])
