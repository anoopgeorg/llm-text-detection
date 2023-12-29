from src.llmTextDetection import logger, logflow
from src.llmTextDetection.entity.config_entity import EvaluationConfig
from src.llmTextDetection.utils.common import save_json
import mlflow
import mlflow.keras
import tensorflow as tf
from pathlib import Path
from urllib.parse import urlparse


class ModelEvaluator:
    @logflow
    def __init__(self, eval_config: EvaluationConfig, runid=None):
        self.eval_config = eval_config
        self.testid = runid
        logger.info("ModelEvaluator instance has be instantiated")

    @logflow
    def evaluateModel(self, model, test_ds):
        if test_ds is not None:
            evaluation = model.evaluate(test_ds)
            self.model = model
            self.scores = {"loss": evaluation[0], "AUC": evaluation[1]}
            save_json(path=Path(self.eval_config.root / self.testid), data=self.scores)

    @logflow
    def logToMlflow(self):
        mlflow.set_registry_uri(self.eval_config.ml_flow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.eval_config.all_params)
            mlflow.log_metrics(self.scores)

            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, registered_model_name="purelyTCNN")
            else:
                mlflow.keras.log_model(self.model, registered_model_name="model")
