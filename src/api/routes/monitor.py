from fastapi import APIRouter
from src.services.monitoring_service import monitoring_service

router = APIRouter()

@router.get("/")
def monitor():
    return monitoring_service.status()