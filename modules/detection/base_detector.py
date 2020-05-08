import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class BaseFaceDetector(ABC):
    """
    Base class for detection model
    """
    def __init__(self, model_path: str, model_config: dict):
        """
        Args:
            model_path: path to model
            model_config: model config
        """
        self.model_path = model_path
        self.model_config = model_config
        self.model = None

    @abstractmethod
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess raw image of RGB format with shape (H, W, C) for model prediction.

        Args:
            image: numpy ndarray image in RGB format. Shape: (H, W, C)

        Returns:
            np.ndarray: preprocessed image for prediction

        """
        pass

    @abstractmethod
    def _predict_raw(self, image: np.ndarray) -> np.ndarray:
        """
        Make prediction on preprocessed image.

        :param image: preprocessed image by _preprocess method.
        :return: raw prediction of model
        """
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method to make prediction on raw image. Return processed detections

        :param image: image in RGB format. Shape: (H, W, C)
        :return: Tuple(np.ndarray, np.ndarray)
            Tuple of bboxes and landmarks
        """
        prep_image = self._preprocess(image)
        raw_output = self.model.predict(prep_image)
        return self._postprocess(raw_output)

    @abstractmethod
    def _postprocess(self, raw_prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method to postprocess model's raw prediction.

        :param raw_prediction: model's raw prediction output
        :return: model's postprocessed output. Tuple of bboxes and landmarks
        """
        pass
