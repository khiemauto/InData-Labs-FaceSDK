import os
import yaml
import numpy as np
import cv2
from sdk import FaceRecognitionSDK


class TestSDK:
    """Test functionality of face recognition SDK."""

    @classmethod
    def setup_class(cls, config_path="config/config.yaml"):

        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        cls.sdk = FaceRecognitionSDK(config)

    def read_image(self, path: str):

        """Reads an image in RGB format."""

        assert os.path.exists(path)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        assert image is not None
        return image

    def save_image(self, image: np.ndarray, path: str):

        """Saves an image in RGB format"""

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, image)

    def test_detector(self):

        image = self.read_image("./test/data/test.jpg")

        bboxes, landmarks = self.sdk.detect_faces(image)

        assert len(bboxes) == len(landmarks) == 5
        # visualize boxes and save image

    def test_alignment(self):

        image = self.read_image("./test/data/test.jpg")

        landmarks = np.array(
            [
                [1126.5667, 306.16245],
                [1245.9434, 301.26337],
                [1169.3936, 383.2747],
                [1142.5924, 418.49274],
                [1260.0977, 413.04846],
            ]
        ).T

        assert landmarks.shape == (2, 5)

        face = self.sdk.align_face(image, landmarks)

        assert face.shape == (112, 112, 3)

        self.save_image(face, "./test/data/alignment_result.jpg")

    def test_embedder(self):

        image = np.random.rand(112, 112, 3).astype(np.float32)
        descriptor = self.sdk.get_descriptor(image)
        assert len(descriptor) == 512
        assert np.allclose(np.power(descriptor, 2).sum(), 1.0)

    def test_database(self):

        image = self.read_image("./test/data/test.jpg")

        bboxes, landmarks = self.sdk.detect_faces(image)

        descriptors = []

        for user_id, (bbox, keypoints) in enumerate(zip(bboxes, landmarks)):

            face = self.sdk.align_face(image, keypoints)
            descriptor = self.sdk.get_descriptor(face)
            descriptors.append(descriptor)
            self.sdk.add_descriptor(descriptor, user_id)

        db_path = "./test/data/test.index"
        self.sdk.save_database(db_path)
        self.sdk.load_database(db_path)
        os.remove(db_path)

        for user_id, descriptor in enumerate(descriptors):

            found_ids, distances = self.sdk.find_most_similar(descriptor, top_k=5)
            assert found_ids[0] == user_id
            assert np.allclose(distances[0], 1.0)
