import time
import gc

from pathlib import Path
from ensure import ensure_annotations


from src.llmTextDetection.components.model_trainer import ModelTrainer
from src.llmTextDetection.pipeline.dataLoader import DataLoader
from src.llmTextDetection.config.configuration import configManager
from src.llmTextDetection import logger, logflow
from src.llmTextDetection.utils.common import savePickle


class ModelTraining:
    def __init__(self, vectorizer_path: Path = None):
        self.runid = f"runid_{time.strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"ModelTraining object instantiated for runid: {self.runid}")
        config_manager = configManager()
        self.model_params = config_manager.getModelParameters()
        self.train_config = config_manager.getTrainerConfig()
        self.data_loader = DataLoader(vectorizer_path=vectorizer_path)
        self.model_trainer = ModelTrainer(
            params=self.model_params, trainer_config=self.train_config, runid=self.runid
        )

    @logflow
    @ensure_annotations
    def trainModel(self):
        # Get the training/validation data and vectorizer
        (
            train_ds,
            train_df,
            validation_ds,
            validation_df,
            vectorizer,
        ) = self.data_loader.getTrainDataset()

        # Save the vectorizer for future
        if train_ds is not None:
            if vectorizer is not None:
                vectorizer_file_path = self.train_config.vectorizer_path / str(
                    self.runid
                )
                # refactor
                vectorizer.save(filepath=str(vectorizer_file_path))

                # save_model(model=vectorizer, filepath=str(vectorizer_file_path))
                # vectorizer.save(str(vectorizer_file_path))
                # tf.saved_model.save(vectorizer, vectorizer_file_path)

                # print("------------++++++++++++||||||||||")
                # print(vectorizer("this"))
                # vectorizer_obj = {
                #     "config": vectorizer.get_config(),
                #     "weights": vectorizer.get_weights(),
                # }
                # savePickle(vectorizer_obj, vectorizer_file_path)
            # Build and begin model training
            # self.model = self.model_trainer.buildModel()
            self.model = self.model_trainer.train(
                train_ds=train_ds,
                train_df=train_df,
                valid_ds=validation_ds,
                valid_df=validation_df,
            )
            # export_model = tf.keras.Sequential(
            #     [vectorizer, self.model, layers.Activation("sigmoid")]
            # )

            # export_model.compile(
            #     optimizer=Adam(), loss=binary_crossentropy, metrics=[AUC(name="auc")]
            # )
            # Save the model for future
            self.model.save(str(self.train_config.model_path / self.runid))
            # export_model.predict(["This is a dummy test"])
            # export_model.save(str(self.train_config.model_path / self.runid))
            del self.model
            del vectorizer
            # del export_model
            gc.collect()


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
