import time
from datetime import datetime

class MonitoringService:
    def __init__(self, model_version: str):
        self.model_version = model_version
        self.start_time = time.time()
        self.inference_count = 0
        self.total_latency = 0.0
        self.last_inference_time = None

    def register_inference(self, latency_ms: float):
        self.inference_count += 1
        self.total_latency += latency_ms
        self.last_inference_time = datetime.utcnow().isoformat()

    def status(self):
        uptime = time.time() - self.start_time
        avg_latency = (
            self.total_latency / self.inference_count
            if self.inference_count > 0
            else 0.0
        )

        return {
            "status": "healthy",
            "model_version": self.model_version,
            "uptime_seconds": round(uptime, 2),
            "inference_count": self.inference_count,
            "avg_latency_ms": round(avg_latency, 2),
            "last_inference_time": self.last_inference_time,
            "service_start_time": datetime.fromtimestamp(self.start_time).isoformat()
        }


monitoring_service = MonitoringService(
    model_version="model_v1"
)
