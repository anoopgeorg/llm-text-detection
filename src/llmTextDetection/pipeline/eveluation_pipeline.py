import pandas as pd
from ensure import ensure_annotations
from pathlib import Path
import time

from src.llmTextDetection.components.evaluator import ModelEvaluator
from src.llmTextDetection.components.predictior import ModelPredictor
from src.llmTextDetection.config.configuration import configManager
from src.llmTextDetection.pipeline.dataLoader import DataLoader
from src.llmTextDetection import logger, logflow


class EvaluationPipeLine:
    def __init__(self, model_path: Path = None, vectorizer_path: Path = None):
        self.config_manager = configManager()
        self.eval_config = self.config_manager.getEvaluationConfig()
        self.pred_config = self.config_manager.getPredictionConfig()

        self.data_loader = DataLoader(vectorizer_path=vectorizer_path)
        self.model_predictor = ModelPredictor(
            prediction_config=self.pred_config,
            model_path=model_path,
            vectorizer=self.data_loader.getVectorizer(),
        )
        self.runid = f"evalid_{time.strftime('%Y%m%d_%H%M%S')}"
        self.evaluator = ModelEvaluator(eval_config=self.eval_config, runid=self.runid)
        self.model = self.model_predictor.model
        self.vectorizer = self.model_predictor.vectorizer
        logger.info("EvaluationPipeLine instance has been instantiated")

    def evaluateModel(self):
        test_ds = self.data_loader.getTestDataset(vectorizer=self.vectorizer)
        self.evaluator.evaluateModel(model=self.model, test_ds=test_ds)
        self.evaluator.logToMlflow()


if __name__ == "__main__":
    STAGE_NAME = "Evaluation"
    try:
        logger.info(f"{STAGE_NAME} stage has started")
        model_training = EvaluationPipeLine()
        model_training.evaluateModel()
        logger.info(f"{STAGE_NAME} stage has completed")
    except Exception as e:
        logger.exception(e)
        raise e
