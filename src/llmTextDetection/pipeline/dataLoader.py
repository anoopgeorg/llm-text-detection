from src.llmTextDetection.config.configuration import configManager
from src.llmTextDetection.components.data_ingestion import DataIngestion
from src.llmTextDetection import logger, logflow

import pandas as pd
import tensorflow as tf
from pathlib import Path
from ensure import ensure_annotations
from keras.layers import TextVectorization


class DataLoader:
    @logflow
    def __init__(self, vectorizer_path: Path = None):
        logger.info(f"Data loader pipeline initiated")
        config_manager = configManager()
        data_ingestion_config = config_manager.getDataIngestionConfig()
        model_paramerters = config_manager.getModelParameters()
        pred_config = config_manager.getPredictionConfig()
        self.data_ingest = DataIngestion(
            config=data_ingestion_config,
            params=model_paramerters,
            pred_config=pred_config,
            vectorizer_path=vectorizer_path,
        )

    @logflow
    @ensure_annotations
    def getTrainDataset(self) -> tuple:
        train = True
        train_df = self.data_ingest.loadData(train=train)
        print(len(train_df))
        train_df = self.data_ingest.getStratifiedData(train_df)
        (
            (train_ds, train_df),
            (validation_ds, validation_df),
            vectorizer,
        ) = self.data_ingest.getDataset(1, train, train_df)

        return (train_ds, train_df, validation_ds, validation_df, vectorizer)

    @logflow
    # @ensure_annotations
    def getTestDataset(
        self, vectorizer: TextVectorization, test_df: pd.DataFrame = None
    ) -> tf.data.Dataset:
        if test_df is None:
            test_df = self.data_ingest.loadData()
        test_ds = self.data_ingest.getDataset(None, False, test_df, vectorizer)
        return test_ds

    def getVectorizer(self):
        return self.data_ingest.loadVectorizer()
