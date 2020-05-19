import numpy as np
import cv2
from typing import List, Tuple

from modules.detection.RetinaFace.model_class import RetinaFace
from modules.recognition.insightface import InsightFaceEmbedder
from modules.alignment import align_and_crop_face
from modules.database import FaissFaceStorage


class FaceRecognitionSDK:
    def __init__(self, config: dict):
        self.detector = RetinaFace(config["detector"])
        self.embedder = InsightFaceEmbedder(config["embedder"])
        self.database = FaissFaceStorage(config["database"])

    def load_database(self, path: str) -> None:
        """Loads database from disk.

        Args:
            path: path to database
        """
        self.database.load(path)

    def save_database(self, path: str) -> None:
        """Saves database to disk.

        Args:
            path: path to database

        """
        self.database.save(path)

    def reset_database(self) -> None:

        self.database.reset()

    def add_photo_by_user_id(self, image: np.ndarray, user_id: int):
        """Adds photo of the user to the database.

        Args:
            image: numpy image (H,W,3) in RGB format.
            user_id: id of the user.
        """

        bboxes, landmarks = self.detect_faces(image)

        if len(bboxes) > 1:
            raise ValueError("Detected more than one face on provided image.")
        elif len(bboxes) == 0:
            raise ValueError("Can't detect any faces on provided image.")

        face = self.align_face(image, landmarks[0])
        descriptor = self.get_descriptor(face)
        self.add_descriptor(descriptor, user_id)

    def add_descriptor(self, descriptor: np.ndarray, user_id: int) -> Tuple[None, int]:

        self.database.add_descriptor(descriptor, user_id)

    def delete_photo_by_id(self, photo_id: int) -> None:
        """Removes photo (descriptor) from the database.

        Args:
            photo_id: id of the photo in the database.

        """
        pass

    def detele_user_by_id(self, user_id: int) -> None:
        """Removes all photos of the user from the database.

        Args:
            user_id: id of the user.
        """
        self.database.remove_user(user_id)

    def find_most_similar(self, descriptor: np.ndarray, top_k: int = 1):
        """Find most similar-looking photos (and their user id's) in the database.

        Args:
            descriptor: descriptor of the photo to use as a search query.
            top_k: number of most similar results to return.
        """
        indicies, distances = self.database.find(descriptor, top_k)
        return indicies, distances

    def verify_faces(self, first_face: np.ndarray, second_face: np.ndarray):
        """Check if two face images are of the same person.

        Args:
            first_face: image of the first face.
            second_face: image of the second face.
        """
        pass

    def detect_faces(self, image: np.ndarray):
        """Detect all faces on the image.

        Args:
            image: numpy image (H,W,3) in RGB format.
        """

        bboxes, landmarks = self.detector.predict(image)
        return bboxes, landmarks

    def recognize_faces(self, image: np.ndarray):
        """Recognize all faces on the image.

        Args:
            image: numpy image (H,W,3) in RGB format.
        """

        bboxes, landmarks = self.detect_faces(image)

        user_ids = []
        similarities = []

        for i, face_keypoints in enumerate(landmarks):

            face = self.align_face(image, face_keypoints)
            descriptor = self.get_descriptor(face)
            indicies, distances = self.find_most_similar(descriptor)
            user_ids.append(indicies[0])
            similarities.append(distances[0])

        return bboxes, landmarks, user_ids, similarities

    def get_descriptor(self, face_image: np.ndarray) -> np.ndarray:
        """Get descriptor of the face image.

        Args:
            face_image: numpy image (112,112,3) in RGB format.

        Returns:
            descriptor: float array of length 512.
        """

        descriptor = self.embedder(face_image)
        return descriptor

    def align_face(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Align face on the image.

        Args:
            image: numpy image (H,W,3) in RGB format.
            landmarks: 5 keypoints of the face to align.
        Returns:
            face: aligned and cropped face image of shape (112,112,3)
        """

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        face = align_and_crop_face(image, landmarks)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        return face

    def set_configuration(self, config: dict):
        """Configure face recognition sdk."""
        pass
