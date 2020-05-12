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

    def test_alignment(self):

        path = "./test/data/test.jpg"
        assert os.path.exists(path)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

        face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        cv2.imwrite("./test/data/alignment_result.jpg", face)

    def test_embedder(self):

        image = np.random.rand(112, 112, 3).astype(np.float32)
        descriptor = self.sdk.get_descriptor(image)
        assert len(descriptor) == 512
        assert np.allclose(np.power(descriptor, 2).sum(), 1.0)
