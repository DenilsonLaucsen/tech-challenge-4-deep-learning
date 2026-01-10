import uvicorn
from fastapi import FastAPI

from src.api.routes.train import router as train_router
from src.api.routes.infer import router as infer_router
from src.api.routes.monitor import router as monitor_router
from src.api.routes.config import router as config_router

app = FastAPI(
    title="ML Training API",
    description="API for training LSTM models",
    version="1.0.0",
)

app.include_router(train_router, prefix="/train", tags=["Training"])
app.include_router(infer_router, prefix="/infer", tags=["Inference"])
app.include_router(monitor_router, prefix="/monitor", tags=["Monitoring"])
app.include_router(config_router, prefix="/config", tags=["Configuration"])


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
