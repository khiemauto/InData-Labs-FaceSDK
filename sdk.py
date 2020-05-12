import numpy as np

from typing import List, Tuple

from modules.detection.RetinaFace.model_class import RetinaFace


class FaceRecognitionSDK:
    def __init__(self, config: dict):
        self.detector = RetinaFace(config["detector"])

    def load_database(self, path: str):
        """Loads database from disk.

        Args:
            path: path to database

        """

        pass

    def save_database(self, path: str):
        """Saves database to disk.

        Args:
            path: path to database

        """

        pass

    def add_photo_by_user_id(self, image: np.ndarray, user_id: int):
        """Adds photo of the user to the database.

        Args:
            image: numpy image (H,W,3) in RGB format.
            user_id: id of the user.
        """

        pass

    def delete_photo_by_id(self, photo_id: int):
        """Removes photo (descriptor) from the database.

        Args:
            photo_id: id of the photo in the database.

        """
        pass

    def detele_user_by_id(self, user_id: int):
        """Removes all photos of the user from the database.

        Args:
            user_id: id of the user.
        """
        pass

    def find_most_similar(self, descriptor: np.ndarray, top_k: int = 1):
        """Find most similar-looking photos (and their user id's) in the database.

        Args:
            descriptor: descriptor of the photo to use as a search query.
            top_k: number of most similar results to return.
        """
        pass

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
        return self.detector.predict(image)

    def recognize_faces(self, image: np.ndarray):
        """Recognize all faces on the image.

        Args:
            image: numpy image (H,W,3) in RGB format.
        """
        pass

    def get_descriptor(self, face_image: np.ndarray):
        """Get descriptor of the face image.

        Args:
            face_image: numpy image (112,112,3) in RGB format.
        """
        pass

    def align_face(self, image: np.ndarray, landmarks: np.ndarray):
        """Align face on the image.

        Args:
            image: numpy image (H,W,3) in RGB format.
            landmarks: 5 keypoints of the face to align.
        """
        pass

    def set_configuration(self, config: dict):
        """Configure face recognition sdk."""
        pass
