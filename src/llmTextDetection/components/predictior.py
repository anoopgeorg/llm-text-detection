from src.llmTextDetection import logger, logflow
from src.llmTextDetection.entity.config_entity import PredictionConfig

from pathlib import Path
from ensure import ensure_annotations
import tensorflow as tf
from tensorflow.keras.models import load_model


class ModelPredictor:
    def __init__(
        self,
        prediction_config: PredictionConfig,
        model_path: Path = None,
        vectorizer_path: Path = None,
    ):
        self.prediction_config = prediction_config
        self.model_path = (
            model_path if model_path is not None else prediction_config.models_root
        )
        self.vectorizer_path = (
            vectorizer_path
            if vectorizer_path is not None
            else prediction_config.vectorizers_root
        )
        self.model = self.loadModel()
        logger.info("ModelPredictor instance has be instantiated")

    @logflow
    @ensure_annotations
    def loadModel(self):
        if self.model_path is not None:
            model = load_model(str(self.model_path))
        return model

    @logflow
    @ensure_annotations
    def loadVectorizer(self):
        if self.vectorizer_path is not None:
            vectorizer = tf.saved_model.load(str(self.vectorizer_path))
        return vectorizer
