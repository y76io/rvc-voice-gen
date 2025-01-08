import firebase_admin
from firebase_admin import credentials, firestore
import os

# Setup credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    "video-dub-e0a3e-firebase-adminsdk-in18v-ec7f75c563.json"
)

import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice_cloning_service")


class FirestoreManager:
    def __init__(self, bucket_name: str, credentials_path: str):
        if not firebase_admin._apps:
            cred = credentials.Certificate(credentials_path)
            initialize_app(cred, {"storageBucket": bucket_name})

        self.db = firestore.Client()

    #     def update_character_model_path(self, project_id, character, model_path):
    #         self.db = firestore.Client()

    #         # Fetch the project document
    #         project_ref = self.db.collection("Projects").document(project_id)
    #         project_data = project_ref.get().to_dict()

    #         if not project_data or "characters" not in project_data:
    #             raise ValueError("Project or characters data not found in Firestore")

    #         # Update the model_path for the matching character
    #         for char in project_data["characters"]:
    #             if char.get("speaker") == character:
    #                 char["model_path"] = model_path

    #         # Update Firestore
    #         project_ref.update({"characters": project_data["characters"]})
    #         logger.info(f"Updated Firestore with model path: {model_path} for character: {character}")

    def update_character_model_path(self, project_id, character, model_path):
        projects_ref = self.db.collection("Projects")
        query = projects_ref.where("name", "==", project_id).limit(1)
        results = query.stream()

        # # Fetch the first result and return the snapshot
        # for doc in results:
        #     project_snapshot = doc  # Firestore document snapshot
        #     break

        for doc in results:
            doc_ref = projects_ref.document(doc.id)  # Create the document reference
            doc_ref, doc  # Return both reference and snapshot
            break

        project_ref = doc_ref
        project_snapshot = doc

        # # Fetch the project document
        # project_ref = self.db.collection("Projects").document(project_id)
        # project_snapshot = project_ref.get()

        if not project_snapshot.exists:
            logger.error(f"Project with ID {project_id} not found in Firestore.")
            raise ValueError(f"Project with ID {project_id} not found in Firestore.")

        project_data = project_snapshot.to_dict()

        if "characters" not in project_data:
            logger.error(f"No 'characters' field found in project {project_id}.")
            raise ValueError(f"No 'characters' field found in project {project_id}.")

        # Update the model_path for the matching character
        character_updated = False
        for char in project_data["characters"]:
            if char.get("speaker") == character:
                char["model_path"] = model_path
                character_updated = True
                break

        if not character_updated:
            logger.error(
                f"Character with speaker '{character}' not found in project {project_id}."
            )
            raise ValueError(
                f"Character with speaker '{character}' not found in project {project_id}."
            )

        # Update Firestore
        project_ref.update({"characters": project_data["characters"]})
        logger.info(
            f"Updated Firestore with model path: {model_path} for character: {character}"
        )
