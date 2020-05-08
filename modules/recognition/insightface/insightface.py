import numpy as np
import torch
from torchvision import transforms

from . import nets

from modules.recognition.base_embedder import BaseFaceEmbedder


class InsightFaceEmbedder(BaseFaceEmbedder):

    """Implements inference of face recognition nets from InsightFace project."""

    def __init__(self, device):

        self.device = device

        self.embedder = nets.iresnet100(pretrained=True).to(device)
        self.embedder.eval()

        mean = [0.5] * 3
        std = [0.5 * 256 / 255] * 3

        self.preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    def _preprocess(self, face: np.ndarray) -> np.ndarray:
        face_tensor = self.preprocess(face).unsqueeze(0).to(self.device)
        return face_tensor

    def _predict_raw(self, face: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            features = self.embedder(face)
        return features

    def _postprocess(self, raw_prediction: np.ndarray) -> np.ndarray:
        descriptor = raw_prediction[0].cpu().numpy()
        descriptor = descriptor / np.linalg.norm(descriptor)
        return descriptor
