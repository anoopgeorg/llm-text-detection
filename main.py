from src.llmTextDetection import logger, logflow
from src.llmTextDetection.pipeline.modelTraining import ModelTraining
from src.llmTextDetection.pipeline.prediction_pipe import PredictionPipeLine
from src.llmTextDetection.pipeline.eveluation_pipeline import EvaluationPipeLine


if __name__ == "__main__":
    STAGE_NAME = "Training"
    try:
        logger.info(f"{STAGE_NAME} stage has started")
        model_training = ModelTraining()
        model_training.trainModel()
        logger.info(f"{STAGE_NAME} stage has completed")
    except Exception as e:
        logger.exception(e)
        raise e

    STAGE_NAME = "Evaluation"
    try:
        logger.info(f"{STAGE_NAME} stage has started")
        model_training = EvaluationPipeLine()
        model_training.evaluateModel()
        logger.info(f"{STAGE_NAME} stage has completed")
    except Exception as e:
        logger.exception(e)
        raise e
