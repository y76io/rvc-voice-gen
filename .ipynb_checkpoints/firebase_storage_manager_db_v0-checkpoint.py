import firebase_admin
from firebase_admin import credentials, storage, initialize_app
from pathlib import Path
import os
from typing import List, Optional
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice_cloning_service")

class CloudStorageManager:
    def __init__(self, bucket_name: str, credentials_path: str):
        """Initialize the Firebase storage manager with a bucket name and credentials.
        
        Args:
            bucket_name: Name of the Firebase Storage bucket (including .appspot.com)
            credentials_path: Path to the service account JSON file
        """
        # Initialize Firebase Admin SDK if not already initialized
        if not firebase_admin._apps:
            cred = credentials.Certificate(credentials_path)
            initialize_app(cred, {
                'storageBucket': bucket_name
            })
        
        # Get bucket instance
        self.bucket = storage.bucket()
        
        # Ensure bucket exists and is accessible
        if not self.bucket.exists():
            raise ValueError(f"Bucket {bucket_name} does not exist or is not accessible")
    
    def _build_audio_path(self, project_id: str, language: str, 
                         character: str, filename: str) -> str:
        """Build the path for audio files."""
        path = f"{project_id}/{language}/{character}/{filename}"# audio/{filename}"
        logger.info(f"Built audio path: {path}")
        return path
    
    def _build_model_path(self, project_id: str, language: str,
                         character: str, model_version: str, filename: str) -> str:
        """Build the path for model files."""
        return f"{project_id}/{language}/{character}/models/{model_version}/{filename}"
    
    def upload_audio(self, local_path: str, project_id: str,
                    language: str, character: str) -> str:
        """Upload an audio file to the specified path structure."""
        try:
            filename = os.path.basename(local_path)
            blob_path = self._build_audio_path(project_id, language, character, filename)
            blob = self.bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            return blob_path
        except Exception as e:
            raise Exception(f"Failed to upload audio file: {str(e)}")
    
    def upload_model(self, local_path: str, project_id: str,
                    language: str, character: str, model_version: str) -> str:
        """Upload a model file to the specified path structure."""
        try:
            filename = os.path.basename(local_path)
            blob_path = self._build_model_path(project_id, language, character, model_version, filename)
            blob = self.bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            return blob_path
        except Exception as e:
            raise Exception(f"Failed to upload model file: {str(e)}")
    
    def download_audio(self, local_dir: str, project_id: str,
                      language: str, character: str, filename: str) -> str:
        """Download an audio file from the specified path."""
        try:
            blob_path = self._build_audio_path(project_id, language, character, filename)
            logger.info(f"Attempting to download from blob path: {blob_path}")

            local_path = os.path.join(local_dir, filename)
            logger.info(f"Attempting to save to local path: {local_path}")

            blob = self.bucket.blob(blob_path)
            logger.info(f"Checking if blob exists: {blob.exists()}")

            if not blob.exists():
                raise Exception(f"Blob does not exist at path: {blob_path}")

            blob.download_to_filename(local_path)
            return local_path
        except Exception as e:
            logger.error(f"""
            Download audio failed with error: {str(e)}
            Project ID: {project_id}
            Language: {language}
            Character: {character}
            Filename: {filename}
            Local Dir: {local_dir}
            """)
            raise
    
    def download_model(self, local_dir: str, project_id: str,
                      language: str, character: str, model_version: str, filename: str) -> str:
        """Download a model file from the specified path."""
        try:
            blob_path = self._build_model_path(project_id, language, character, model_version, filename)
            local_path = os.path.join(local_dir, filename)
            blob = self.bucket.blob(blob_path)
            blob.download_to_filename(local_path)
            return local_path
        except Exception as e:
            raise Exception(f"Failed to download model file: {str(e)}")
    
    def list_audio_files(self, project_id: str, language: str, character: str) -> List[str]:
        """List all audio files in the specified path."""
        try:
            prefix = f"{project_id}/{language}/{character}/" # audio/"
            blobs = self.bucket.list_blobs(prefix=prefix)
            return [blob.name.split('/')[-1] for blob in blobs]
        except Exception as e:
            raise Exception(f"Failed to list audio files: {str(e)}")
    
    def list_model_files(self, project_id: str, language: str,
                        character: str, model_version: Optional[str] = None) -> List[str]:
        """List all model files in the specified path."""
        try:
            prefix = f"{project_id}/{language}/{character}/models/"
            if model_version:
                prefix += f"{model_version}/"
            blobs = self.bucket.list_blobs(prefix=prefix)
            return [blob.name.split('/')[-1] for blob in blobs]
        except Exception as e:
            raise Exception(f"Failed to list model files: {str(e)}")