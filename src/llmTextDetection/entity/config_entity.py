from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    raw_train_data_path: Path
    raw_test_data_path: Path
    pre_processing_path: Path


@dataclass(frozen=True)
class ModelParameters:
    class_names: list
    num_classes: int
    classes: list
    class_2_name: dict
    num_folds: int
    seed: int
    max_sequence: int
    max_tokens: int
    batch_size: int
    selected_folds: list
    ml_lr: float
    embd_dim: int
    epochs: int
    device: str
    strategy: tf.distribute.Strategy


@dataclass(frozen=True)
class TrainerConfig:
    root_dir: Path
    vectorizer_path: Path
    model_path: Path


@dataclass(frozen=True)
class PredictionConfig:
    models_root: Path
    vectorizers_root: Path


@dataclass(frozen=True)
class EvaluationConfig:
    root: Path
    ml_flow_uri: str
    ml_flow_uname: str
    ml_flow_pass: str
    all_params: dict
    eval_model_path: Path
