from fastapi import APIRouter, HTTPException
from pathlib import Path

import yaml

router = APIRouter()

@router.get("/")
def get_config():
    try:
        config_path = Path("configs/best_config.yaml")

        if not config_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Champion config not found. Run champion_selector.py first.",
            )
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
