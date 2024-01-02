import pandas as pd
import re
import os
import gc
from keras.models import load_model
from src.llmTextDetection.utils.common import find_latest_file
from ensure import ensure_annotations
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from keras.layers import TextVectorization


from src.llmTextDetection import logger, logflow
from src.llmTextDetection.utils.common import savePickle, loadPickle
from src.llmTextDetection.entity.config_entity import (
    DataIngestionConfig,
    ModelParameters,
    PredictionConfig,
)


class DataIngestion:
    def __init__(
        self,
        config: DataIngestionConfig,
        params: ModelParameters,
        pred_config: PredictionConfig,
        vectorizer_path: Path = None,
    ):
        self.config = config
        self.params = params
        self.pred_config = pred_config
        self.vectorizer_path = vectorizer_path
        logger.info("DataIngestion object initialized")

    @logflow
    @ensure_annotations
    def getData(self, path: Path) -> pd.DataFrame:
        data = []
        try:
            logger.info(f"CWD ====={os.getcwd()}")
            # Collect all the training files from raw data
            for file_path in path.iterdir():
                df = pd.read_csv(file_path)
                data.append(df)
            df = pd.concat(data)
            return df
        except Exception as e:
            logger.info(e)

    @logflow
    @ensure_annotations
    def loadData(self, train: bool = False) -> pd.DataFrame:
        try:
            if train:
                train_df = self.getData(Path(self.config.raw_train_data_path))
                train_df["label_name"] = train_df["label"].map(self.params.class_2_name)
                return train_df
            else:
                test_df = self.getData(Path(self.config.raw_test_data_path))
                return test_df
        except Exception as e:
            logger.info(e)

    @logflow
    @ensure_annotations
    def getStratifiedData(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Create a stratification(label,source),fold column
            df["stratify"] = df["label"].astype(str) + df["source"].astype(str)
            df["fold"] = 0

            skf = StratifiedKFold(
                n_splits=self.params.num_folds,
                random_state=self.params.seed,
                shuffle=True,
            )
            # assign folds by index
            for fold, (train_inx, valid_inx) in enumerate(
                skf.split(df, df["stratify"])
            ):
                df.loc[valid_inx, "fold"] = fold
            return df
        except Exception as e:
            logger.info(e)

    @logflow
    @ensure_annotations
    def getRegexExclusions(self, df: pd.DataFrame) -> tuple:
        # Get list of charcters missing from the human data
        human_df = df[df["label_name"] == self.params.class_names[0]].copy()  # Human
        ai_df = df[df["label_name"] == self.params.class_names[1]].copy()  # AI

        human_characters = set("".join(human_df["text"].to_list()))
        ai_characters = set("".join(ai_df["text"].to_list()))

        caracter_exclusion = "".join(
            [x for x in ai_characters if x not in human_characters]
        )
        # logger.info(caracter_exclusion)
        caracter_escape = re.escape(caracter_exclusion)
        char_excl_regex = f"[{caracter_escape}]"

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

        return (char_excl_regex, regex_pattern, html_pattern)

    def standardizeText(self, input_data):
        if (
            self.regex_pattern is not None
            and self.html_exclude is not None
            and self.html_exclude is not None
        ):
            data = tf.strings.lower(input_data)
            data = tf.strings.regex_replace(data, self.regex_pattern, "")
            data = tf.strings.regex_replace(data, self.html_exclude, "")
            data = tf.strings.regex_replace(data, self.char_excl_regex, "")
            return data

    @logflow
    @ensure_annotations
    # Preprocessing and vectorization of text
    def buildVectorizationLayer(self, texts: list, df: pd.DataFrame):
        # Get Regex patterns
        (
            self.char_excl_regex,
            self.regex_pattern,
            self.html_exclude,
        ) = self.getRegexExclusions(df)
        preprocess_artifacts = {
            "char_excl_regex": self.char_excl_regex,
            "regex_pattern": self.regex_pattern,
            "html_exclude": self.html_exclude,
        }
        save_path = str(self.config.pre_processing_path / "regex_patterns.pkl")
        savePickle(
            obj=preprocess_artifacts,
            path=save_path,
        )

        # tf.keras.utils.get_custom_objects()["standardizeText"] = self.standardizeText
        vectorization_layer = TextVectorization(
            # standardize="standardizeText",
            max_tokens=self.params.max_tokens,
            output_mode="int",
            output_sequence_length=self.params.max_sequence,
            input_shape=(1,),
        )
        train_text = tf.data.Dataset.from_tensor_slices(texts)

        logger.info("====>text preprocess started for training data has started")
        processed_text = train_text.map(self.standardizeText)
        logger.info("====>text preprocess started for training data has ended")
        logger.info("====>Vocabulary adaption for training data has started")
        vectorization_layer.adapt(processed_text)
        logger.info("<====Vocabulary adaption for training data has ended")
        # refactor
        vectorizer_model = tf.keras.models.Sequential([vectorization_layer])
        vectorizer_model(tf.constant(["test"], dtype=tf.string))
        vectorizer_model.summary()
        vectorizer_model.compile()
        del vectorization_layer
        gc.collect()
        # return vectorization_layer
        return vectorizer_model

    def loadVectorizer(self):
        # tf.keras.utils.get_custom_objects()["standardizeText"] = self.standardizeText
        path = (
            str(self.vectorizer_path)
            if self.vectorizer_path is not None
            else str(find_latest_file(self.pred_config.vectorizers_root))
        )
        vectorizer = load_model(path)
        return vectorizer

    @logflow
    @ensure_annotations
    def buildDataset(
        self,
        texts,
        labels=None,
        batch_size=32,
        shuffle=False,
        drop_remainder=True,
        repeat=False,
        vectorizer=None,
        # pre_process=False,
    ):
        AUTO = tf.data.AUTOTUNE
        slices = (texts) if labels is None else (texts, labels)
        ds = tf.data.Dataset.from_tensor_slices(slices)

        # Vectorization function
        def vectorizeText(texts, labels=None):
            texts = vectorizer(texts)
            return (texts) if labels is None else (texts, labels)

        # if pre_process:
        #     regex_patterns = loadPickle(
        #         str(self.config.pre_processing_path / "regex_patterns.pkl")
        #     )
        #     (self.char_excl_regex, self.regex_pattern, self.html_exclude) = (
        #         regex_patterns["char_excl_regex"],
        #         regex_patterns["regex_pattern"],
        #         regex_patterns["html_exclude"],
        #     )

        #     logger.info("====>text preprocess started for training data has started")
        #     ds = ds.map(self.standardizeText, num_parallel_calls=AUTO)
        #     logger.info("====>text preprocess started for training data has ended")
        ds = ds.map(vectorizeText, num_parallel_calls=AUTO)

        ds = ds.repeat() if repeat else ds
        opt = tf.data.Options()
        if shuffle:
            ds = ds.shuffle(shuffle, seed=self.params.seed)
            opt.experimental_deterministic = False
        ds = ds.with_options(opt)
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        ds = ds.prefetch(AUTO)
        return ds

    def buildTestDataset(
        self,
        texts,
        labels=None,
        batch_size=32,
        shuffle=False,
        drop_remainder=True,
        repeat=False,
        vectorizer=None,
    ):
        AUTO = tf.data.AUTOTUNE
        texts = tf.data.Dataset.from_tensor_slices((texts))
        labels = tf.data.Dataset.from_tensor_slices((labels))
        regex_patterns = loadPickle(
            str(self.config.pre_processing_path / "regex_patterns.pkl")
        )
        (self.char_excl_regex, self.regex_pattern, self.html_exclude) = (
            regex_patterns["char_excl_regex"],
            regex_patterns["regex_pattern"],
            regex_patterns["html_exclude"],
        )

        logger.info("====>text preprocess started for training data has started")
        texts = texts.map(self.standardizeText, num_parallel_calls=AUTO)
        logger.info("====>text preprocess started for training data has ended")

        # slices = (texts) if labels is None else (texts, labels)
        # ds = tf.data.Dataset.from_tensor_slices(slices)
        ds = texts if labels is None else tf.data.Dataset.zip((texts, labels))

        # Vectorization function
        def vectorizeText(texts, labels=None):
            texts = vectorizer(texts)
            return (texts) if labels is None else (texts, labels)

        ds = ds.map(vectorizeText, num_parallel_calls=AUTO)

        ds = ds.repeat() if repeat else ds
        opt = tf.data.Options()
        if shuffle:
            ds = ds.shuffle(shuffle, seed=self.params.seed)
            opt.experimental_deterministic = False
        ds = ds.with_options(opt)
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        ds = ds.prefetch(AUTO)
        return ds

    @logflow
    def getDataset(self, fold=None, train=False, df=None, vectorizer=None):
        if train == True:  # create training and validation set
            if fold in df["fold"].value_counts().index.to_list():
                # Create the training dataset
                train_df = df[df["fold"] != fold].sample(frac=1)
                train_text = train_df["text"].to_list()
                train_labels = train_df["label"].to_list()

                # Vectorize the text based on training data
                vectorizer = self.buildVectorizationLayer(train_text, train_df)

                train_ds = self.buildDataset(
                    train_text,
                    train_labels,
                    batch_size=self.params.batch_size,
                    shuffle=True,
                    drop_remainder=True,
                    repeat=True,
                    vectorizer=vectorizer,
                )
                # Create the validation dataset

                valid_df = df[df["fold"] == fold].sample(frac=1)
                valid_text = valid_df["text"].to_list()
                valid_labels = valid_df["label"].to_list()

                valid_ds = self.buildDataset(
                    valid_text,
                    valid_labels,
                    batch_size=self.params.batch_size,
                    shuffle=False,
                    drop_remainder=True,
                    repeat=False,
                    vectorizer=vectorizer,
                )

                return (train_ds, train_df), (valid_ds, valid_df), vectorizer
            else:
                logger.info(f"Selected fold:{fold} unavailable in data frame")
        else:  # Create test data set
            if vectorizer is not None:
                test_text = df["text"].astype("str").to_list()
                test_labels = df["label"].to_list()
                # Vectorize the text based on training data
                # test_ds = self.buildDataset(
                #     test_text,
                #     test_labels,
                #     batch_size=self.params.batch_size,
                #     shuffle=False,
                #     drop_remainder=False,
                #     repeat=False,
                #     vectorizer=vectorizer,
                #     # pre_process=True,
                # )
                test_ds = self.buildTestDataset(
                    test_text,
                    test_labels,
                    batch_size=self.params.batch_size,
                    shuffle=False,
                    drop_remainder=False,
                    repeat=False,
                    vectorizer=vectorizer,
                )
                return test_ds
