import tensorflow as tf
import re
from tensorflow.config import list_logical_devices


from src.llmTextDetection.constants import *
from src.llmTextDetection.utils.common import read_yaml, create_directory, savePickle
from src.llmTextDetection import logger, logflow
from src.llmTextDetection.entity.config_entity import (
    DataIngestionConfig,
    ModelParameters,
    TrainerConfig,
    PredictionConfig,
    EvaluationConfig,
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
                self.config.data_ingestion.pre_processing_path,
            ]
        )
        # Create default pre-processing regex patterns
        # Regex pattern for excluding emojis
        exclude = re.compile(
            "["
            "\u000A"  # new-line
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002500-\U00002BEF"  # chinese char
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"  # dingbats
            "\u3030"
            "]+",
            re.UNICODE,
        )
        regex_pattern = exclude.pattern

        # Regex pattern for excluding html tags
        html_exclude = re.compile(r"<.*?>")
        html_pattern = html_exclude.pattern
        preprocess_artifacts = {
            "regex_pattern": regex_pattern,
            "html_exclude": html_pattern,
        }
        savepath = str(Path(config.default_pre_process))
        savePickle(
            obj=preprocess_artifacts,
            path=savepath,
        )
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            raw_train_data_path=config.raw_train_data_path,
            raw_test_data_path=config.raw_test_data_path,
            pre_processing_path=Path(config.pre_processing_path),
            default_pre_process=Path(config.default_pre_process),
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
        params = self.params
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
            models_root=Path(config.models_root),
            vectorizers_root=Path(config.vectorizers_root),
            default_vectorizer=Path(config.default_vectorizer),
            default_model=Path(config.default_model),
        )
        return predictor_config

    @logflow
    def getEvaluationConfig(self) -> EvaluationConfig:
        config = self.config.evaluator
        create_directory([config.root])
        evaluator_config = EvaluationConfig(
            root=Path(config.root),
            ml_flow_uri=config.ml_flow_uri,
            ml_flow_uname=config.ml_flow_uname,
            ml_flow_pass=config.ml_flow_pass,
            all_params=self.params,
            scores_file=config.scores_file,
            eval_model_path=Path(config.eval_model_path),
        )
        return evaluator_config
