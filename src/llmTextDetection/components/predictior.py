from src.llmTextDetection import logger, logflow
from src.llmTextDetection.entity.config_entity import PredictionConfig
from src.llmTextDetection.utils.common import find_latest_file, loadPickle

from pathlib import Path
from ensure import ensure_annotations
import tensorflow as tf
from keras.models import load_model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.metrics import AUC


class ModelPredictor:
    def __init__(
        self,
        prediction_config: PredictionConfig,
        model_path: Path = None,
        vectorizer=None,
    ):
        self.prediction_config = prediction_config
        self.model = None
        self.model_path = model_path
        self.model = self.loadModel()
        self.vectorizer = vectorizer
        logger.info("ModelPredictor instance has be instantiated")

    @logflow
    @ensure_annotations
    def loadModel(self):
        if self.model_path is not None:
            model = load_model(str(self.model_path))
        else:
            model = load_model(
                str(find_latest_file(self.prediction_config.models_root))
            )
        # print("XXXXXXXXX Check this" * 3)
        model.summary()
        model.compile(
            optimizer=Adam(), loss=binary_crossentropy, metrics=[AUC(name="auc")]
        )
        return model
