import time
from ensure import ensure_annotations
from keras.layers import (
    Dense,
    LSTM,
    TextVectorization,
    Embedding,
    BatchNormalization,
    Input,
    Conv1D,
    MaxPool1D,
    Flatten,
    Dropout,
)
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.metrics import AUC
import tensorflow as tf
import pandas as pd

from src.llmTextDetection import logger, logflow
from src.llmTextDetection.entity.config_entity import (
    DataIngestionConfig,
    ModelParameters,
)


class ModelTrainer:
    def __init__(self, params: ModelParameters):
        self.ruind = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"ModelTrainer initialized with ruind: {self.ruind} ")
        self.params = params

    @logflow
    @ensure_annotations
    def buildModel(self) -> Model:
        inp = Input(shape=(self.params.max_sequence,))
        embed = Embedding(
            input_dim=self.params.max_tokens,
            output_dim=self.params.embd_dim,
            input_shape=(self.params.max_sequence,),
        )(inp)
        conv1d = Conv1D(filters=128, kernel_size=7)(embed)
        maxpool = MaxPool1D(pool_size=2, strides=2)(conv1d)
        conv1d_2 = Conv1D(filters=32, kernel_size=4)(maxpool)
        maxpool_2 = MaxPool1D(pool_size=2, strides=2)(conv1d_2)
        flatten = Flatten()(maxpool_2)
        drop = Dropout(0.3)(flatten)
        dense = Dense(units=16, activation="relu")(drop)
        out = Dense(units=1, activation="sigmoid")(dense)

        model = Model(inputs=inp, outputs=out)
        model.summary()
        model.compile(
            optimizer=Adam(), loss=binary_crossentropy, metrics=[AUC(name="auc")]
        )
        return model

    @logflow
    @ensure_annotations
    def trainModel(
        self,
        model: Model,
        train_ds: tf.data.Dataset,
        train_df: pd.DataFrame,
        valid_ds: tf.data.Dataset,
        valid_df: pd.DataFrame,
    ):
        with self.params.strategy.scope():
            model = self.buildModel()
            model.fit(
                train_ds,
                epochs=self.params.epochs,
                validation_data=valid_ds,
                steps_per_epoch=(len(train_df) // self.params.batch_size),
                validation_steps=(len(valid_df) // self.params.batch_size),
            )
