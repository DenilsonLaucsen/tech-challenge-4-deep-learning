from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union, Optional
from datetime import datetime
import time

from src.inference.predictor import predict
from src.services.monitoring_service import monitoring_service

router = APIRouter()

class InferRequest(BaseModel):
    sequence: List[Union[float, List[float]]] = Field(..., description="Janela temporal. Ex: [[f1,f2,...], ...] ou [v1,v2,...]")

class InferResponse(BaseModel):
    prediction: float
    timestamp: datetime
    latency_ms: float

@router.post("/", response_model=InferResponse, tags=["Inference"])
def infer_endpoint(req: InferRequest):
    start_time = time.perf_counter()
    try:
        result = predict(
            sequence=req.sequence,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro de inferência: {e}")
    
    latency_ms = (time.perf_counter() - start_time) * 1000

    monitoring_service.register_inference(latency_ms)

    return InferResponse(
        prediction=result["prediction"],
        timestamp=datetime.utcnow(),
        latency_ms=latency_ms,
    )
