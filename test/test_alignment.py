import cv2
import numpy as np
import os

from sdk import FaceRecognitionSDK


def test_alignment():

    sdk = FaceRecognitionSDK()

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

    face = sdk.align_face(image, landmarks)

    assert face.shape == (112, 112, 3)

    face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./test/data/alignment_result.jpg", face)
