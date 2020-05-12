from modules.recognition.insightface import InsightFaceEmbedder
import numpy as np


def test_insightface():

    embedder = InsightFaceEmbedder(device="cpu")
    image = np.random.rand(112, 112, 3).astype(np.float32)
    descriptor = embedder(image)
    assert len(descriptor) == 512
    assert np.allclose(np.power(descriptor, 2).sum(), 1.0)
