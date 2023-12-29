import pandas as pd
from ensure import ensure_annotations
from pathlib import Path


from src.llmTextDetection.components.predictior import ModelPredictor
from src.llmTextDetection.config.configuration import configManager
from src.llmTextDetection import logger, logflow
from src.llmTextDetection.pipeline.dataLoader import DataLoader


class PredictionPipeLine:
    def __init__(self, model_path: Path = None, vectorizer_path: Path = None):
        self.config_manager = configManager()
        self.prediction_config = self.config_manager.getPredictionConfig()
        self.data_loader = DataLoader(vectorizer_path=vectorizer_path)
        self.model_predictor = ModelPredictor(
            prediction_config=self.prediction_config,
            model_path=model_path,
            vectorizer=self.data_loader.getVectorizer(),
        )
        self.model = self.model_predictor.loadModel()
        self.vectorizer = self.model_predictor.vectorizer
        logger.info("PredictionPipeLine instance has been instantiated")

    @logflow
    @ensure_annotations
    def makePrediction(self, test_df: pd.DataFrame = None):
        if self.vectorizer is not None:
            test_ds = self.data_loader.getTestDataset(
                vectorizer=self.vectorizer, test_df=test_df
            )
            predictions = self.model.predict(test_ds)
            return predictions
        else:
            logger.exception("Vectorizer not loaded")


if __name__ == "__main__":
    STAGE_NAME = "Prediction"
    try:
        logger.info(f"{STAGE_NAME} stage has started")
        model_training = PredictionPipeLine()
        model_training.makePrediction()
        logger.info(f"{STAGE_NAME} stage has completed")
    except Exception as e:
        logger.exception(e)
        raise e
