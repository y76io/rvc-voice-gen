import logging
from typing import Optional, List
from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_server")

app = FastAPI(title="Test Client Server", version="1.0")

MAIN_SERVER_URL = "http://localhost:8000"
RECEIVED_CALLBACKS = []  # store callbacks received from main server

class TestTrainRequest(BaseModel):
    client_id: str = "test_client"
    project_id: str = "test_project"
    character: str = "test_character"
    language: str = "en"
    model_name: str = "TestModel"
    callback_url: Optional[str] = "http://localhost:9000/callback"
    total_epoch: int = 5

class TestInferRequest(BaseModel):
    client_id: str = "test_client"
    project_id: str = "test_project"
    character: str = "test_character"
    language: str = "en"
    model_version: str = "en_2024_01_01"
    model_name: str = "TestModel"
    input_filenames: List[str] = ["input_audio.wav"]
    callback_url: Optional[str] = "http://localhost:9000/callback"
    f0_up_key: int = 5
    export_format: str = "WAV"

@app.post("/test_train")
def test_train(req: TestTrainRequest):
    payload = req.dict()
    # You may add other params if you want to test them
    payload["rvc_version"] = "v2"
    payload["index_algorithm"] = "Auto"
    logger.info(f"Sending training request to main server: {payload}")
    resp = requests.post(f"{MAIN_SERVER_URL}/train", json=payload)
    return {"status_code": resp.status_code, "response": resp.json()}

@app.post("/test_infer")
def test_infer(req: TestInferRequest):
    payload = req.dict()
    logger.info(f"Sending inference request to main server: {payload}")
    resp = requests.post(f"{MAIN_SERVER_URL}/infer", json=payload)
    return {"status_code": resp.status_code, "response": resp.json()}

@app.post("/callback")
async def callback_endpoint(request: Request):
    data = await request.json()
    logger.info(f"Received callback from main server: {data}")
    RECEIVED_CALLBACKS.append(data)
    return {"status": "ok"}

@app.get("/callbacks")
def get_callbacks():
    return {"callbacks": RECEIVED_CALLBACKS}
