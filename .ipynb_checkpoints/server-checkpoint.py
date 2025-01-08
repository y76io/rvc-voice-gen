import os
import logging
import datetime
import requests
import shutil
from typing import Optional, List
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path

from firebase_storage_manager import CloudStorageManager

# from cloud_storage_manager import CloudStorageManager
# from train import preprocess_dataset, extract_features, train_model, create_index, download_model as train_download_model
# from inference import run_inference

from rvc_training import (
    preprocess_dataset,
    extract_features,
    train_model,
    create_index,
    download_model as train_download_model,
)
from rvc_inference import run_inference


# -----------------------
# Configuration
# -----------------------
BUCKET_NAME = os.environ.get("CLOUD_BUCKET_NAME", "video-dub-e0a3e.appspot.com")
CREDENTIALS_PATH = os.environ.get(
    "FIREBASE_CREDENTIALS", "video-dub-e0a3e-firebase-adminsdk-in18v-ec7f75c563.json"
)
LOCAL_DATA_DIR = Path("data")
LOCAL_LOGS_DIR = Path("logs")
LOCAL_MODEL_CACHE_MAX = 20  # max number of models to keep locally
LOCAL_MODELS_DIR = LOCAL_LOGS_DIR  # models are stored in logs/<model_name>
LOCAL_OUTPUT_DIR = Path("outputs")  # for inference outputs
LOCAL_DATA_DIR.mkdir(exist_ok=True)
LOCAL_OUTPUT_DIR.mkdir(exist_ok=True)
LOCAL_LOGS_DIR.mkdir(exist_ok=True)

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice_cloning_service")

# -----------------------
# Initialize Cloud Storage Manager
# -----------------------
# storage_manager = CloudStorageManager(bucket_name=BUCKET_NAME)
storage_manager = CloudStorageManager(
    bucket_name=BUCKET_NAME, credentials_path=CREDENTIALS_PATH
)


# -----------------------
# Utility Functions
# -----------------------
def prune_model_cache():
    """Ensure that the local model cache does not exceed LOCAL_MODEL_CACHE_MAX."""
    model_dirs = [d for d in LOCAL_LOGS_DIR.iterdir() if d.is_dir()]
    if len(model_dirs) <= LOCAL_MODEL_CACHE_MAX:
        return
    # Sort by modification time, oldest first
    model_dirs.sort(key=lambda d: d.stat().st_mtime)
    to_remove = len(model_dirs) - LOCAL_MODEL_CACHE_MAX
    for i in range(to_remove):
        shutil.rmtree(model_dirs[i])
        logger.info(f"Removed old model cache directory: {model_dirs[i].name}")


def model_exists_locally(model_name: str) -> bool:
    model_path = LOCAL_LOGS_DIR / model_name
    return model_path.exists()


def download_model_from_cloud(
    client_id: str, project_id: str, character: str, model_version: str, model_name: str
):
    """Download model files from cloud storage to local logs directory."""
    target_dir = LOCAL_LOGS_DIR / model_name
    target_dir.mkdir(exist_ok=True, parents=True)

    model_files = storage_manager.list_model_files(
        client_id, project_id, character, model_version=model_version
    )
    for mf in model_files:
        storage_manager.download_model(
            str(target_dir), client_id, project_id, character, model_version, mf
        )
    logger.info(f"Downloaded model {model_name} ({model_version}) from cloud storage.")
    prune_model_cache()


def upload_model_to_cloud(
    client_id: str, project_id: str, character: str, model_version: str, model_name: str
):
    """Upload trained model files to cloud storage."""
    model_path = LOCAL_LOGS_DIR / model_name
    for f in model_path.iterdir():
        if f.is_file():
            storage_manager.upload_model(
                str(f), client_id, project_id, character, model_version
            )
    logger.info(f"Uploaded model {model_name} ({model_version}) to cloud storage.")


def download_all_audio_files(
    client_id: str, project_id: str, character: str, language: str
) -> Path:
    """Download all audio files for the dataset from cloud storage."""
    dataset_dir = LOCAL_DATA_DIR / f"{client_id}_{project_id}_{character}_{language}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    audio_files = storage_manager.list_audio_files(
        client_id, project_id, character, language
    )
    for af in audio_files:
        storage_manager.download_audio(
            str(dataset_dir), client_id, project_id, character, language, af
        )
    logger.info(
        f"Downloaded {len(audio_files)} audio files for {client_id}/{project_id}/{character}/{language}."
    )
    return dataset_dir


def upload_inference_results(
    output_paths: List[str],
    client_id: str,
    project_id: str,
    character: str,
    language: str,
) -> List[str]:
    """Upload inference results to cloud storage and return the blob paths."""
    uploaded_paths = []
    for p in output_paths:
        blob_path = storage_manager.upload_audio(
            p, client_id, project_id, character, language
        )
        uploaded_paths.append(blob_path)
    logger.info(f"Uploaded {len(output_paths)} inference results to cloud storage.")
    return uploaded_paths


def send_callback(webhook_url: str, data: dict):
    """Send a callback to the provided webhook URL with given data."""
    try:
        resp = requests.post(webhook_url, json=data, timeout=10)
        logger.info(
            f"Callback POST to {webhook_url} completed with status {resp.status_code}."
        )
    except Exception as e:
        logger.error(f"Failed to send callback to {webhook_url}: {e}")


# -----------------------
# Request Models with full parameters
# -----------------------


class TrainRequest(BaseModel):
    # Identifying info
    client_id: str
    project_id: str
    character: str
    language: str
    model_name: str
    callback_url: Optional[str] = None

    # Preprocessing parameters
    dataset_sample_rate: int = 40000
    preprocess_cpu_cores: int = 2
    cut_preprocess: bool = True
    process_effects: bool = False
    noise_reduction: bool = False
    noise_reduction_strength: float = 0.7

    # Feature extraction parameters
    rvc_version: str = "v2"
    f0_method: str = "rmvpe"
    hop_length: int = 128
    extract_cpu_cores: int = 2
    extract_gpu: int = 0
    embedder_model: str = "contentvec"
    embedder_model_custom: str = ""

    # Training parameters
    total_epoch: int = 20
    batch_size: int = 15
    gpu: int = 0
    pretrained: bool = True
    custom_pretrained: bool = False
    g_pretrained_path: str = ""
    d_pretrained_path: str = ""
    overtraining_detector: bool = False
    overtraining_threshold: int = 50
    cleanup: bool = False
    cache_data_in_gpu: bool = False
    save_every_epoch: int = 10
    save_only_latest: bool = False
    save_every_weights: bool = False
    index_algorithm: str = "Auto"


class InferenceRequest(BaseModel):
    # Identifying info
    client_id: str
    project_id: str
    character: str
    language: str
    model_version: str
    model_name: str
    input_filenames: List[str]
    callback_url: Optional[str] = None

    # Inference parameters (default values from previous code)
    export_format: str = "WAV"
    f0_method: str = "rmvpe"
    f0_up_key: int = 0
    filter_radius: int = 3
    rms_mix_rate: float = 0.8
    protect: float = 0.5
    index_rate: float = 0.7
    hop_length: int = 128
    clean_strength: float = 0.7
    split_audio: bool = False
    clean_audio: bool = False
    f0_autotune: bool = False
    formant_shift: bool = False
    formant_qfrency: float = 1.0
    formant_timbre: float = 1.0
    embedder_model: str = "contentvec"
    embedder_model_custom: str = ""

    # Post-processing effects
    post_process: bool = False
    reverb: bool = False
    pitch_shift: bool = False
    limiter: bool = False
    gain: bool = False
    distortion: bool = False
    chorus: bool = False
    bitcrush: bool = False
    clipping: bool = False
    compressor: bool = False
    delay: bool = False

    reverb_room_size: float = 0.5
    reverb_damping: float = 0.5
    reverb_wet_gain: float = 0.0
    reverb_dry_gain: float = 0.0
    reverb_width: float = 1.0
    reverb_freeze_mode: float = 0.0

    pitch_shift_semitones: float = 0.0

    limiter_threshold: float = -1.0
    limiter_release_time: float = 0.05

    gain_db: float = 0.0

    distortion_gain: float = 0.0

    chorus_rate: float = 1.5
    chorus_depth: float = 0.1
    chorus_center_delay: float = 15.0
    chorus_feedback: float = 0.25
    chorus_mix: float = 0.5

    bitcrush_bit_depth: int = 4

    clipping_threshold: float = 0.5

    compressor_threshold: float = -20.0
    compressor_ratio: float = 4.0
    compressor_attack: float = 0.001
    compressor_release: float = 0.1

    delay_seconds: float = 0.1
    delay_feedback: float = 0.5
    delay_mix: float = 0.5


# -----------------------
# Server Initialization
# -----------------------
app = FastAPI(title="Voice Cloning Service", version="1.0")

# -----------------------
# Endpoints
# -----------------------


@app.post("/train")
def train_endpoint(req: TrainRequest, background_tasks: BackgroundTasks):
    print(f"Received training request: {req}")
    logger.info(f"Received training request: {req}")
    model_version = f"{req.language}_{datetime.date.today().strftime('%Y_%m_%d')}"
    dataset_dir = download_all_audio_files(
        req.client_id, req.project_id, req.character, req.language
    )

    def run_training_task():
        try:
            # Preprocess
            preprocess_dataset(
                model_name=req.model_name,
                dataset_path=str(dataset_dir),
                sample_rate=req.dataset_sample_rate,
                cpu_cores=req.preprocess_cpu_cores,
                cut_preprocess=req.cut_preprocess,
                process_effects=req.process_effects,
                noise_reduction=req.noise_reduction,
                noise_reduction_strength=req.noise_reduction_strength,
            )
            # Extract
            extract_features(
                model_name=req.model_name,
                rvc_version=req.rvc_version,
                f0_method=req.f0_method,
                hop_length=req.hop_length,
                sample_rate=req.dataset_sample_rate,
                cpu_cores=req.extract_cpu_cores,
                gpu=req.extract_gpu,
                embedder_model=req.embedder_model,
                embedder_model_custom=req.embedder_model_custom,
            )
            # Train
            train_model(
                model_name=req.model_name,
                rvc_version=req.rvc_version,
                total_epoch=req.total_epoch,
                sample_rate=req.dataset_sample_rate,
                batch_size=req.batch_size,
                gpu=req.gpu,
                pretrained=req.pretrained,
                custom_pretrained=req.custom_pretrained,
                g_pretrained_path=req.g_pretrained_path,
                d_pretrained_path=req.d_pretrained_path,
                overtraining_detector=req.overtraining_detector,
                overtraining_threshold=req.overtraining_threshold,
                cleanup=req.cleanup,
                cache_data_in_gpu=req.cache_data_in_gpu,
                save_every_epoch=req.save_every_epoch,
                save_only_latest=req.save_only_latest,
                save_every_weights=req.save_every_weights,
            )
            # Index
            create_index(
                req.model_name,
                rvc_version=req.rvc_version,
                index_algorithm=req.index_algorithm,
            )
            # Upload model
            upload_model_to_cloud(
                req.client_id,
                req.project_id,
                req.character,
                model_version,
                req.model_name,
            )
            if req.callback_url:
                send_callback(
                    req.callback_url,
                    {
                        "status": "completed",
                        "message": f"Training complete for {req.model_name}",
                        "model_version": model_version,
                    },
                )
        except Exception as e:
            logger.error(f"Training task failed: {e}")
            if req.callback_url:
                send_callback(
                    req.callback_url,
                    {
                        "status": "error",
                        "message": f"Training failed for {req.model_name}: {str(e)}",
                    },
                )

    background_tasks.add_task(run_training_task)
    return {
        "status": "accepted",
        "message": f"Training started for {req.model_name}",
        "results_location": f"{req.client_id}/{req.project_id}/{req.character}/models/{model_version}/",
    }


@app.post("/infer")
def inference_endpoint(req: InferenceRequest, background_tasks: BackgroundTasks):
    logger.info(f"Received inference request: {req}")
    # Ensure model is cached locally
    if not model_exists_locally(req.model_name):
        logger.info(f"Model {req.model_name} not found locally, downloading...")
        download_model_from_cloud(
            req.client_id,
            req.project_id,
            req.character,
            req.model_version,
            req.model_name,
        )
    else:
        logger.info(f"Model {req.model_name} found locally.")

    # Download input files
    input_paths = []
    input_dir = (
        LOCAL_DATA_DIR
        / f"infer_{req.client_id}_{req.project_id}_{req.character}_{req.language}"
    )
    input_dir.mkdir(exist_ok=True, parents=True)
    for fn in req.input_filenames:
        local_path = storage_manager.download_audio(
            str(input_dir),
            req.client_id,
            req.project_id,
            req.character,
            req.language,
            fn,
        )
        input_paths.append(local_path)

    def run_inference_task():
        try:
            output_paths = []
            for inp in input_paths:
                output_path = str(
                    LOCAL_OUTPUT_DIR
                    / (Path(inp).stem + "_converted." + req.export_format.lower())
                )
                run_inference(
                    model_name=req.model_name,
                    input_path=inp,
                    output_path=output_path,
                    export_format=req.export_format,
                    f0_method=req.f0_method,
                    f0_up_key=req.f0_up_key,
                    filter_radius=req.filter_radius,
                    rms_mix_rate=req.rms_mix_rate,
                    protect=req.protect,
                    index_rate=req.index_rate,
                    hop_length=req.hop_length,
                    clean_strength=req.clean_strength,
                    split_audio=req.split_audio,
                    clean_audio=req.clean_audio,
                    f0_autotune=req.f0_autotune,
                    formant_shift=req.formant_shift,
                    formant_qfrency=req.formant_qfrency,
                    formant_timbre=req.formant_timbre,
                    embedder_model=req.embedder_model,
                    embedder_model_custom=req.embedder_model_custom,
                    post_process=req.post_process,
                    reverb=req.reverb,
                    pitch_shift=req.pitch_shift,
                    limiter=req.limiter,
                    gain=req.gain,
                    distortion=req.distortion,
                    chorus=req.chorus,
                    bitcrush=req.bitcrush,
                    clipping=req.clipping,
                    compressor=req.compressor,
                    delay=req.delay,
                    reverb_room_size=req.reverb_room_size,
                    reverb_damping=req.reverb_damping,
                    reverb_wet_gain=req.reverb_wet_gain,
                    reverb_dry_gain=req.reverb_dry_gain,
                    reverb_width=req.reverb_width,
                    reverb_freeze_mode=req.reverb_freeze_mode,
                    pitch_shift_semitones=req.pitch_shift_semitones,
                    limiter_threshold=req.limiter_threshold,
                    limiter_release_time=req.limiter_release_time,
                    gain_db=req.gain_db,
                    distortion_gain=req.distortion_gain,
                    chorus_rate=req.chorus_rate,
                    chorus_depth=req.chorus_depth,
                    chorus_center_delay=req.chorus_center_delay,
                    chorus_feedback=req.chorus_feedback,
                    chorus_mix=req.chorus_mix,
                    bitcrush_bit_depth=req.bitcrush_bit_depth,
                    clipping_threshold=req.clipping_threshold,
                    compressor_threshold=req.compressor_threshold,
                    compressor_ratio=req.compressor_ratio,
                    compressor_attack=req.compressor_attack,
                    compressor_release=req.compressor_release,
                    delay_seconds=req.delay_seconds,
                    delay_feedback=req.delay_feedback,
                    delay_mix=req.delay_mix,
                )
                output_paths.append(output_path)

            uploaded_files = upload_inference_results(
                output_paths, req.client_id, req.project_id, req.character, req.language
            )
            if req.callback_url:
                send_callback(
                    req.callback_url,
                    {
                        "status": "completed",
                        "message": "Inference completed",
                        "uploaded_files": uploaded_files,
                    },
                )
        except Exception as e:
            logger.error(f"Inference task failed: {e}")
            if req.callback_url:
                send_callback(
                    req.callback_url,
                    {"status": "error", "message": f"Inference failed: {str(e)}"},
                )

    background_tasks.add_task(run_inference_task)
    return {
        "status": "accepted",
        "message": "Inference task started",
        "results_location": f"{req.client_id}/{req.project_id}/{req.character}/{req.language}/audio/",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
