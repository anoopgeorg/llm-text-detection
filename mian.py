from src.llmTextDetection.config.configuration import configManager
from src.llmTextDetection.components.data_ingestion import DataIngestion
from src.llmTextDetection.constants import *


if __name__ == "__main__":
    config_manager = configManager()
    data_ingestion_config = config_manager.getDataIngestionConfig()
    model_paramerters = config_manager.getModelParameters()
    test = DataIngestion(config=data_ingestion_config, params=model_paramerters)
    (train_df, test_df) = test.loadData()
    print(len(train_df))
    print(len(test_df))
    test_ds = test.getDataset(None, True, test_df, vectorizer)
