import os
import logging
import datetime
import requests
import shutil
from typing import Optional, List
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path

from firebase_storage_manager_db_v0 import CloudStorageManager
from firestore_manager import FirestoreManager  # contains firestore util functions
from rvc_training import (
    preprocess_dataset,
    extract_features,
    train_model,
    create_index,
    download_model as train_download_model,
)
from rvc_inference import run_inference


# Add these imports at the top of server_db_v0.py
from fastapi import Request, HTTPException, Query
from io import BytesIO
import requests
import json
from pydub import AudioSegment
import tempfile

import asyncio
from pathlib import Path
import os
import tempfile
import logging

from typing import Optional

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
storage_manager = CloudStorageManager(
    bucket_name=BUCKET_NAME, credentials_path=CREDENTIALS_PATH
)

firestore_manager = FirestoreManager(
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
    project_id: str, language: str, character: str, model_version: str, model_name: str
):
    """Download model files from cloud storage to local logs directory."""
    target_dir = LOCAL_LOGS_DIR / model_name
    target_dir.mkdir(exist_ok=True, parents=True)

    model_files = storage_manager.list_model_files(
        project_id, language, character, model_version=model_version
    )
    for mf in model_files:
        storage_manager.download_model(
            str(target_dir), project_id, language, character, model_version, mf
        )
    logger.info(f"Downloaded model {model_name} ({model_version}) from cloud storage.")
    prune_model_cache()


def upload_model_to_cloud(
    project_id: str, language: str, character: str, model_version: str, model_name: str
):
    """Upload trained model files to cloud storage."""
    model_path = LOCAL_LOGS_DIR / model_name
    for f in model_path.iterdir():
        if f.is_file():
            storage_manager.upload_model(
                str(f), project_id, language, character, model_version
            )
    logger.info(f"Uploaded model {model_name} ({model_version}) to cloud storage.")


def download_audio_files(
    project_id: str, language: str, character: str, filenames: List[str]
) -> Path:
    """Download specific audio files for the dataset from cloud storage."""
    try:
        dataset_dir = LOCAL_DATA_DIR / f"{project_id}_{language}_{character}"
        logger.info(f"Creating dataset directory at: {dataset_dir}")
        dataset_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"""
        Starting download of {len(filenames)} files:
        Project ID: {project_id}
        Language: {language}
        Character: {character}
        First few filenames: {filenames[:3]}
        Target directory: {dataset_dir}
        """
        )

        for i, filename in enumerate(filenames):
            try:
                logger.info(f"Downloading file {i+1}/{len(filenames)}: {filename}")
                storage_manager.download_audio(
                    str(dataset_dir), project_id, language, character, filename
                )
            except Exception as e:
                logger.error(f"Failed to download file {filename}: {str(e)}")
                raise

        logger.info(
            f"Successfully downloaded {len(filenames)} audio files to {dataset_dir}"
        )
        return dataset_dir
    except Exception as e:
        logger.error(f"Failed in download_audio_files: {str(e)}")
        raise


def upload_inference_results(
    output_paths: List[str], project_id: str, language: str, character: str
) -> List[str]:
    """Upload inference results to cloud storage and return the blob paths."""
    uploaded_paths = []
    for p in output_paths:
        blob_path = storage_manager.upload_audio(p, project_id, language, character)
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
# Request Models
# -----------------------
class TrainRequest(BaseModel):
    # Identifying info
    # client_id: str  # Kept for backwards compatibility but not used in paths
    project_id: str
    character: str
    language: str
    model_name: str
    audio_filenames: List[str]  # List of audio files to use for training
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
    total_epoch: int = 50  # 20
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
    # client_id: str  # Kept for backwards compatibility but not used in paths
    project_id: str
    character: str
    language: str
    model_version: str
    model_name: str
    input_filenames: List[str]
    callback_url: Optional[str] = None

    # Inference parameters
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

    print("Debug - Request details:")
    print(f"Project ID: {req.project_id}")
    print(f"Language: {req.language}")
    print(f"Character: {req.character}")
    print(f"Audio filenames type: {type(req.audio_filenames)}")
    print(f"Audio filenames: {req.audio_filenames}")

    if not req.audio_filenames:
        return {"status": "error", "message": "No audio files provided for training"}

    model_version = f"{req.language}_{datetime.date.today().strftime('%Y_%m_%d')}"
    dataset_dir = download_audio_files(
        req.project_id, req.language, req.character, req.audio_filenames
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

            # Extract features
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

            # Create index
            create_index(
                req.model_name,
                rvc_version=req.rvc_version,
                index_algorithm=req.index_algorithm,
            )

            # Upload model
            upload_model_to_cloud(
                req.project_id,
                req.language,
                req.character,
                model_version,
                req.model_name,
            )

            model_path_storage = (
                f"{req.language}/{req.character}/{model_version}/{req.model_name}"
            )
            logging.info(
                f"upload model_path: {model_path_storage}, project_id: {req.project_id}, character: {req.character}"
            )
            firestore_manager.update_character_model_path(
                req.project_id, req.character, model_path_storage
            )
            # firestore_manager.update_character_model_path(project_id, character, model_path)

            # if req.callback_url:
            # send_callback(req.callback_url, {
            #     "status": "completed",
            #     "message": f"Training complete for {req.model_name}",
            #     "project_id": req.project_id,
            #     "language": req.language,
            #     "character": req.character,
            #     "model_version": model_version,
            #     "model_name": req.model_name
            # })
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
        "results_location": f"{req.project_id}/{req.language}/{req.character}/models/{model_version}/",
    }


@app.post("/infer")
def inference_endpoint(req: InferenceRequest, background_tasks: BackgroundTasks):
    logger.info(f"Received inference request: {req}")

    # Ensure model is cached locally
    if not model_exists_locally(req.model_name):
        logger.info(f"Model {req.model_name} not found locally, downloading...")
        download_model_from_cloud(
            req.project_id,
            req.language,
            req.character,
            req.model_version,
            req.model_name,
        )
    else:
        logger.info(f"Model {req.model_name} found locally.")

    # Download input files
    input_paths = req.input_filenames  # []
    input_dir = (
        LOCAL_DATA_DIR / f"infer_{req.project_id}_{req.language}_{req.character}"
    )
    # input_dir.mkdir(exist_ok=True, parents=True)
    # for fn in req.input_filenames:
    #     local_path = storage_manager.download_audio(str(input_dir), req.project_id, req.language, req.character, fn)
    #     input_paths.append(local_path)

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

            ### handle the uploading outside the inference function
            # logger.info(f"output_paths: {output_paths}")
            # uploaded_files = upload_inference_results(output_paths, req.project_id, req.language, req.character)
            # if req.callback_url:
            #     send_callback(req.callback_url, {
            #         "status": "completed",
            #         "message": "Inference completed",
            #         "uploaded_files": uploaded_files
            #     })

            logger.info(f" inference output_paths: {output_paths}")

            return output_paths
        except Exception as e:
            logger.error(f"Inference task failed: {e}")
            return None
            # if req.callback_url:
            #     send_callback(req.callback_url, {
            #         "status": "error",
            #         "message": f"Inference failed: {str(e)}"
            #     })

    # background_tasks.add_task(run_inference_task)
    output_paths = run_inference_task()
    return output_paths
    # return {
    #     "status": "accepted",
    #     "message": "Inference task started",
    #     "results_location": f"{req.project_id}/{req.language}/{req.character}/" #audio/"
    # }


# Additions


from firebase_admin import firestore
from fastapi import HTTPException
from typing import List, Dict


async def fetch_training_data_from_firestore(
    project_id: str, callback_url: str
) -> List[TrainRequest]:
    """
    Fetches project data from Firestore and constructs training requests for each character

    Args:
        project_id: The ID of the project in Firestore

    Returns:
        List of TrainRequest objects for each character with sufficient audio samples
    """
    try:
        # Initialize Firestore client
        db = firestore.client()

        # Get project document
        project_doc = db.collection("Projects").document(project_id).get()
        if not project_doc.exists:
            raise HTTPException(
                status_code=404, message=f"Project {project_id} not found"
            )

        project_data = project_doc.to_dict()
        characters = project_data.get("characters", [])
        project_name = project_data.get("name")
        original_lang = project_data.get("orginalLang")

        if not characters:
            raise HTTPException(
                status_code=400, message="No characters found in project"
            )

        training_requests = []

        # Process each character
        for speaker in characters:
            # Query high-quality audio samples
            dialogue_ref = (
                db.collection("Projects")
                .document(project_id)
                .collection("Dialogue")
                .where("originalAudioRating", "in", [4, 5])
                .where("speaker", "==", speaker.get("speaker"))
                .stream()
            )

            # Collect audio files
            audio_files = []
            for doc in dialogue_ref:
                doc_data = doc.to_dict()
                if original_audio := doc_data.get("originalAudio"):
                    # Extract filename from URL
                    filename = original_audio.split("/")[-1]
                    audio_files.append(filename)

            # Skip if insufficient samples
            if len(audio_files) < 1:  # Change to 10 in production
                logger.warning(
                    f"Skipping {speaker.get('speaker')}: insufficient samples ({len(audio_files)})"
                )
                continue

            # Limit to 30 samples
            audio_files = audio_files[:30]

            # Create training request
            model_name = f"{speaker.get('speaker')}_{original_lang}"

            train_request = TrainRequest(
                project_id=project_name,
                character=speaker.get("speaker"),
                language=original_lang,
                model_name=model_name,
                audio_filenames=audio_files,
                callback_url=callback_url,
                total_epoch=100,
                # Using default values for all other training parameters
            )

            training_requests.append(train_request)

        return training_requests

    except Exception as e:
        logger.error(f"Error fetching training data from Firestore: {str(e)}")
        raise HTTPException(
            status_code=500, message=f"Failed to fetch training data: {str(e)}"
        )


@app.post("/train-from-firestore/{project_id}")
async def train_from_firestore(
    project_id: str,
    background_tasks: BackgroundTasks,
    callback_url: Optional[str] = Query(
        "http://localhost:4001/api/services/callBackQueueUpdateVoiceTraining"
    ),
):

    # @app.route('/train-from-firestore', methods=['POST'])
    # def train_from_firestore():
    #     data = request.get_json()
    #     project_id = data['project_id'] #data['url']

    """
    Endpoint that fetches training data from Firestore and initiates training for all eligible characters
    """
    try:
        # Fetch training requests
        training_requests = await fetch_training_data_from_firestore(
            project_id, callback_url
        )

        if not training_requests:
            return {
                "status": "error",
                "message": "No eligible characters found for training",
            }

        # Start training for each character
        results = []
        for train_req in training_requests:
            result = train_endpoint(train_req, background_tasks)
            results.append(
                {
                    "character": train_req.character,
                    "status": result["status"],
                    "message": result["message"],
                    "results_location": result.get("results_location"),
                }
            )

        return {
            "status": "accepted",
            "message": f"Training initiated for {len(results)} characters",
            "results": results,
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in train_from_firestore: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Add these new request models
class DialogueData(BaseModel):
    speaker: str
    start: float
    end: float
    translatedText: str
    voiceId: str
    voiceId2: Optional[str] = None
    languageCode: str


class VoiceConversionRequest(BaseModel):
    flag: int = 0
    dialogue: DialogueData
    elevenLabKey: str
    baseDirectory: str
    silence_thresh: float = -40
    padding: int = 80


# Add these utility functions
def trim_leading_trailing_silence(
    audio_segment: AudioSegment, silence_thresh: float = -40, padding_ms: int = 80
) -> tuple:
    """Trim silence from the beginning and end of the audio."""

    def detect_leading_silence(sound):
        trim_ms = 0
        chunk_size = 10
        while (
            trim_ms < len(sound)
            and sound[trim_ms : trim_ms + chunk_size].dBFS < silence_thresh
        ):
            trim_ms += chunk_size
        return trim_ms

    start_trim = detect_leading_silence(audio_segment)
    end_trim = len(audio_segment) - detect_leading_silence(audio_segment.reverse())

    # Add padding
    start_trim = max(0, start_trim - padding_ms)
    end_trim = min(len(audio_segment), end_trim + padding_ms)

    return audio_segment[start_trim:end_trim], start_trim, end_trim


def generate_tts(text: str, voice_id: str, api_key: str) -> AudioSegment:
    """Generate text-to-speech audio using ElevenLabs API"""
    response = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        json={
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5,
            },
        },
        headers={
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key,
        },
        stream=True,
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate audio: {response.text}"
        )

    return AudioSegment.from_file(BytesIO(response.content), format="mp3")


def download_model_data(
    project_id: str, language: str, character: str, model_name: str, model_version: str
):
    # logger.info(f"Received inference request: {req}")

    # Ensure model is cached locally
    if not model_exists_locally(model_name):
        logger.info(f"Model {model_name} not found locally, downloading...")
        download_model_from_cloud(
            project_id, language, character, model_version, model_name
        )
    else:
        logger.info(f"Model {model_name} found locally.")

    # # Download input files
    # input_paths = []
    # input_dir = LOCAL_DATA_DIR / f"infer_{project_id}_{language}_{character}"
    # input_dir.mkdir(exist_ok=True, parents=True)
    # for fn in input_filenames:
    #     local_path = storage_manager.download_audio(str(input_dir), project_id, language, character, fn)
    #     input_paths.append(local_path)


async def convert_to_voice(
    audio_segment: AudioSegment,
    project_id: str,
    language: str,
    character: str,
    model_name: str,
    model_version: str,
) -> AudioSegment:
    """Convert audio using local inference endpoint instead of ElevenLabs"""
    temp_filename = None
    try:
        download_model_data(project_id, language, character, model_name, model_version)

        # Verify model files exist
        model_dir = (
            LOCAL_LOGS_DIR / model_name
        )  # This will be logs/SPEAKER_00_en_20e_80s/
        logger.info(f"Checking model directory: {model_dir}")

        if not model_dir.exists():
            raise ValueError(f"Model directory not found: {model_dir}")

        # List existing files in model directory
        existing_files = list(model_dir.glob("*"))
        logger.info(f"Files found in model directory: {existing_files}")

        # Check for required model files
        # Note: model_version parameter might be the .pth file name
        model_files = list(model_dir.glob("G_*.pth"))
        if not model_files:
            raise ValueError(f"No model weights file found in {model_dir}")

        #### the inference function already handles the .index file and .npy file
        # index_file = model_dir / "added.index"
        # if not index_file.exists():
        #     raise ValueError(f"Index file not found: {index_file}")

        # feature_file = model_dir / "total_fea.npy"
        # if not feature_file.exists():
        #     raise ValueError(f"Feature file not found: {feature_file}")

        logger.info(f"Found all required model files in {model_dir}")

        # # Rest of the function remains the same...
        # # Save audio to temporary file
        # with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        #     audio_segment.export(temp_file.name, format="wav")
        #     temp_filename = temp_file.name

        ### modified here

        import tempfile
        import uuid

        # save the input audio to local path
        input_dir = LOCAL_DATA_DIR / f"infer_{project_id}_{language}_{character}"
        logger.info(f"input_dir is {input_dir}")
        input_dir.mkdir(exist_ok=True, parents=True)

        # with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        random_name = f"temp_{uuid.uuid4().hex}.wav"
        logger.info(f"random_name is {random_name}")
        temp_filename = os.path.join(input_dir, random_name)  # temp_file.name
        audio_segment.export(temp_filename, format="wav")
        logger.info(f"saved temp file to path {temp_filename}")

        logger.info(f"Saved temporary audio file: {temp_filename}")

        # Create inference request with correct model information
        inference_req = InferenceRequest(
            project_id=project_id,
            character=character,
            language=language,
            model_version=model_version,  # This should be the .pth file name
            model_name=model_name,  # This should be the directory name
            input_filenames=[temp_filename],
            f0_method="rmvpe",
            protect=0.5,
            clean_audio=True,
        )

        ##### no need to upload the temp file then download it
        # # Upload the temporary file to storage
        # blob_path = storage_manager.upload_audio(temp_filename, project_id, language, character)
        # logger.info(f"Uploaded audio to: {blob_path}")

        # Create background tasks handler
        class SyncBackgroundTasks:
            def __init__(self):
                self.tasks = []
                self.errors = []

            def add_task(self, func, *args, **kwargs):
                self.tasks.append((func, args, kwargs))

            def run_all(self):
                for task, args, kwargs in self.tasks:
                    try:
                        task(*args, **kwargs)
                    except Exception as e:
                        self.errors.append(str(e))
                if self.errors:
                    raise Exception(
                        f"Background tasks failed: {'; '.join(self.errors)}"
                    )

        # Run inference
        background_tasks = SyncBackgroundTasks()
        # response = inference_endpoint(inference_req, background_tasks)
        output_paths = inference_endpoint(inference_req, background_tasks)

        logger.info(f" convert_to_voice, output_paths: {output_paths}")

        #         if response["status"] != "accepted":
        #             raise ValueError(f"Inference failed: {response.get('message', 'Unknown error')}")

        #         logger.info(f"Inference response: {response}")
        background_tasks.run_all()

        return output_paths

    #         # Wait for processing
    #         await asyncio.sleep(5)

    #         # Check for output file
    #         output_filename = f"{os.path.basename(temp_filename).rsplit('.', 1)[0]}_converted.wav"
    #         output_blob = storage_manager.bucket.blob(f"{project_id}/{language}/{character}/{output_filename}")

    #         retry_count = 0
    #         max_retries = 3
    #         while retry_count < max_retries:
    #             if output_blob.exists():
    #                 break
    #             retry_count += 1
    #             logger.info(f"Output file not ready, waiting... (attempt {retry_count}/{max_retries})")
    #             await asyncio.sleep(2)

    #         if not output_blob.exists():
    #             raise FileNotFoundError(f"Output file never appeared in storage: {output_filename}")

    #         # Download converted file
    #         converted_path = storage_manager.download_audio(
    #             str(LOCAL_OUTPUT_DIR),
    #             project_id,
    #             language,
    #             character,
    #             output_filename
    #         )

    #         logger.info(f"Downloaded converted audio to: {converted_path}")
    #         return AudioSegment.from_wav(converted_path)

    except Exception as e:
        logger.error(f"Error in convert_to_voice: {str(e)}", exc_info=True)
        raise


##### uncomment to clean temp files
#     finally:
#         if temp_filename and os.path.exists(temp_filename):
#             try:
#                 # os.unlink(temp_filename) ## uncomment
#                 logger.info("remove temp files")
#             except Exception as e:
#                 logger.warning(f"Failed to cleanup temporary file {temp_filename}: {str(e)}")


# Add the new endpoint
@app.post("/convert_voice")
async def convert_voice(request: VoiceConversionRequest):
    try:
        dialogue = request.dialogue

        if request.flag not in [0, 1]:
            raise HTTPException(status_code=400, detail="Invalid flag value")

        # Case 1: Regular text-to-speech (flag = 0)
        if request.flag == 0:
            audio_content = generate_tts(
                text=dialogue.translatedText,
                voice_id=dialogue.voiceId,
                api_key=request.elevenLabKey,
            )

            trimmed_audio, start_trim, end_trim = trim_leading_trailing_silence(
                audio_content, request.silence_thresh, request.padding
            )

            trimmed_from_start_secs = start_trim / 1000
            trimmed_from_end_secs = (len(audio_content) - end_trim) / 1000

        # Case 2: Text-to-speech then voice conversion (flag = 1)
        elif request.flag == 1:
            if not dialogue.voiceId2:
                raise HTTPException(
                    status_code=400, detail="Missing second voice ID for conversion"
                )

            # First generate audio with ElevenLabs
            initial_audio = generate_tts(
                text=dialogue.translatedText,
                voice_id=dialogue.voiceId,
                api_key=request.elevenLabKey,
            )

            # Trim silence
            trimmed_audio, start_trim, end_trim = trim_leading_trailing_silence(
                initial_audio, request.silence_thresh, request.padding
            )

            # Extract model info from voiceId2
            # Format should be: "project_id/language/character/model_version/weights_file"
            voice_parts = dialogue.voiceId2.split("/")
            if len(voice_parts) != 5:
                raise HTTPException(status_code=400, detail="Invalid voiceId2 format")

            project_id, language, character, model_version, weights_file = voice_parts

            # Construct the model name from components
            # The model directory name should match the base name of the weights file
            model_name = weights_file.replace(".pth", "")
            if model_name.startswith("G_"):
                model_name = model_name[2:]  # Remove 'G_' prefix if present

            logger.info(
                f"""Model parameters:
                Project ID: {project_id}
                Language: {language}
                Character: {character}
                Model Name: {model_name}
                Model Version: {model_version}
                Weights File: {weights_file}
            """
            )

            # Convert using local inference
            output_paths = await convert_to_voice(
                trimmed_audio,
                project_id,
                language,
                character,
                model_name,
                model_version,
            )

            trimmed_from_start_secs = start_trim / 1000
            trimmed_from_end_secs = (len(initial_audio) - end_trim) / 1000

        #         # # Generate filename
        #         # filenamePrefix = f"{dialogue.speaker}/{dialogue.start}_{dialogue.end}_{dialogue.speaker}"
        #         # timestamp = datetime.datetime.now().isoformat().replace(':', '_').replace('.', '_')
        #         # filename = f"{request.baseDirectory}/{filenamePrefix}_{timestamp}.wav"

        #         # Save to a bytes buffer to upload
        #         trimmed_buffer = BytesIO()
        #         # audio = AudioSegment.from_file(audio_path, format="wav")
        #         trimmed_audio.export(trimmed_buffer, format="wav")
        #         trimmed_buffer.seek(0)

        #         audio = AudioSegment.from_file(output_paths[0], format="wav")

        #         # Upload to storage
        #         blob = storage_manager.bucket.blob(filename)
        #         blob.upload_from_file(trimmed_buffer, content_type='audio/wav')
        #         blob.make_public()
        #         public_url = blob.public_url

        # dialogue.languageCode

        # dialogue

        # Generate filename
        filenamePrefix = f"{dialogue.speaker}/{dialogue.languageCode}/{dialogue.start}_{dialogue.end}_{dialogue.speaker}"
        timestamp = (
            datetime.datetime.now().isoformat().replace(":", "_").replace(".", "_")
        )
        filename = f"{request.baseDirectory}/{filenamePrefix}_{timestamp}.wav"

        # audio = AudioSegment.from_file(ouptut_paths[0], format="wav")
        # print("Audio loaded successfully (optional step)")

        logger.info(f"upload file, output_paths: {output_paths}")
        # Upload to storage directly from file
        blob = storage_manager.bucket.blob(filename)
        blob.upload_from_filename(output_paths[0], content_type="audio/wav")
        blob.make_public()

        logger.info("file uploaded")
        # Return the public URL of the uploaded file
        public_url = blob.public_url

        logger.info(
            f"filename: {filename}, url: {public_url}, audio_length: {len(trimmed_audio) / 1000}, trimmed_start: {trimmed_from_start_secs}, trimmed_end: {trimmed_from_end_secs}"
        )

        return {
            "filename": filename,
            "url": public_url,
            "audio_length": len(trimmed_audio) / 1000,
            "trimmed_start": trimmed_from_start_secs,
            "trimmed_end": trimmed_from_end_secs,
        }

        # return jsonify({
        #     "filename": filename,
        #     "url": public_url,
        #     "audio_length": len(trimmed_audio) / 1000,  # Length in seconds
        #     "trimmed_start": trimmed_from_start_secs,  # In seconds
        #     "trimmed_end": trimmed_from_end_secs,  # In seconds
        # }), 200

    except Exception as e:
        logger.error(f"Error in convert_voice: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
