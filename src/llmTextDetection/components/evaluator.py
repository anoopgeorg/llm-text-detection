from src.llmTextDetection import logger, logflow
from src.llmTextDetection.entity.config_entity import EvaluationConfig
from src.llmTextDetection.utils.common import save_score

import os
import mlflow
import mlflow.keras
import tensorflow as tf
from pathlib import Path
from urllib.parse import urlparse
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.metrics import AUC


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
            self.scores = {self.testid: {"loss": evaluation[0], "AUC": evaluation[1]}}
            save_score(
                path=Path(self.eval_config.scores_file),
                data=self.scores,
            )

    @logflow
    def logToMlflow(self):
        os.environ["MLFLOW_TRACKING_URI"] = self.eval_config.ml_flow_uri
        os.environ["MLFLOW_TRACKING_USERNAME"] = self.eval_config.ml_flow_uname
        os.environ["MLFLOW_TRACKING_PASSWORD"] = self.eval_config.ml_flow_pass
        print(
            os.environ["MLFLOW_TRACKING_URI"],
            os.environ["MLFLOW_TRACKING_USERNAME"],
            os.environ["MLFLOW_TRACKING_PASSWORD"],
        )
        mlflow.set_tracking_uri(self.eval_config.ml_flow_uri)
        mlflow.set_registry_uri(self.eval_config.ml_flow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.eval_config.all_params)
            mlflow.log_metrics(self.scores[self.testid])

            artifact_path = str(self.eval_config.eval_model_path)
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(
                    self.model,
                    registered_model_name="purelyTCNN",
                    artifact_path=artifact_path,
                )
            else:
                mlflow.keras.log_model(
                    self.model,
                    registered_model_name="model",
                    artifact_path=artifact_path,
                )
