import tensorflow as tf
from tensorflow.config import list_logical_devices


from src.llmTextDetection.constants import *
from src.llmTextDetection.utils.common import read_yaml, create_directory
from src.llmTextDetection import logger, logflow
from src.llmTextDetection.entity.config_entity import (
    DataIngestionConfig,
    ModelParameters,
    TrainerConfig,
    PredictionConfig,
)


class configManager:
    def __init__(
        self, config_file_path=CONFIG_FILE_PATH, params_file_path=PARAMS_FILE_PATH
    ):
        """ """
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        create_directory([self.config.artifacts_root])

    @logflow
    def getDataIngestionConfig(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directory(
            [
                self.config.data_ingestion.root_dir,
                self.config.data_ingestion.train_data_path,
                self.config.data_ingestion.test_data_path,
            ]
        )
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            raw_train_data_path=config.raw_train_data_path,
            raw_test_data_path=config.raw_test_data_path,
        )
        return data_ingestion_config

    @logflow
    def getDevice(self) -> (tf.distribute.Strategy, str):
        gpus = list_logical_devices("GPU")
        ngpu = len(gpus)
        # Check number of GPUs
        if ngpu:
            # Set GPU strategy
            strategy = tf.distribute.MirroredStrategy(gpus)  # single-GPU or multi-GPU
            # Print GPU details
            logger.info("> Running on GPU")
            logger.info(f"Num of GPUs:{ngpu} ")
            device = "GPU"
        else:
            # If no GPUs are available, use CPU
            logger.info("> Running on CPU")
            strategy = tf.distribute.get_strategy()
            device = "CPU"
        return (strategy, device)

    @logflow
    def getModelParameters(self) -> ModelParameters:
        params = self.params.model_paramerters
        classes = list(range(params.num_classes))
        strategy, device = self.getDevice()
        model_parameters = ModelParameters(
            class_names=params.class_names,
            num_classes=params.num_classes,
            classes=classes,
            class_2_name=dict(zip(classes, params.class_names)),
            num_folds=params.num_folds,
            seed=params.seed,
            max_sequence=params.max_sequence,
            max_tokens=params.max_tokens,
            batch_size=params.batch_size,
            selected_folds=params.selected_folds,
            ml_lr=params.ml_lr,
            embd_dim=params.embd_dim,
            epochs=params.epochs,
            device=device,
            strategy=strategy,
        )
        return model_parameters

    @logflow
    def getTrainerConfig(self) -> TrainerConfig:
        config = self.config.model_trainer
        create_directory(
            [
                config.root_dir,
                config.model_path,
                config.vectorizer_path,
            ]
        )
        trainer_config = TrainerConfig(
            model_path=Path(config.model_path),
            root_dir=Path(config.root_dir),
            vectorizer_path=Path(config.vectorizer_path),
        )
        return trainer_config

    @logflow
    def getPredictionConfig(self) -> PredictionConfig:
        config = self.config.predictor
        predictor_config = PredictionConfig(
            models_root=config.models_root, vectorizers_root=config.vectorizers_root
        )
        return predictor_config
