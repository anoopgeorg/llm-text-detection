import pandas as pd
import re
import os
from ensure import ensure_annotations
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
import tensorflow as tf
from keras.layers import TextVectorization


from src.llmTextDetection import logger, logflow
from src.llmTextDetection.entity.config_entity import (
    DataIngestionConfig,
    ModelParameters,
)


class DataIngestion:
    def __init__(self, config: DataIngestionConfig, params: ModelParameters):
        self.config = config
        self.params = params
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

            skf = StratifiedGroupKFold(
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
    def getRegexExclusions(self, df: pd.DataFrame) -> (str, re.Pattern, re.Pattern):
        # Get list of charcters missing from the human data
        human_df = df[df["label_name"] == self.params.class_names[0]].copy()  # Human
        ai_df = df[df["label_name"] == self.params.class_names[1]].copy()  # AI

        human_characters = set("".join(human_df["text"].to_list()))
        ai_characters = set("".join(ai_df["text"].to_list()))

        caracter_exclusion = "".join(
            [x for x in ai_characters if x not in human_characters]
        )
        logger.info(caracter_exclusion)
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

        return (char_excl_regex, regex_pattern, html_exclude)

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
    def buildVectorizationLayer(self, texts: pd.DataFrame):
        # Get Regex patterns
        (
            self.char_excl_regex,
            self.regex_pattern,
            self.html_exclude,
        ) = self.getRegexExclusions(texts)

        vectorization_layer = TextVectorization(
            standardize=self.standardizeText,
            max_tokens=self.params.max_tokens,
            output_mode="int",
            output_sequence_length=self.params.max_sequence,
        )
        train_text = tf.data.Dataset.from_tensor_slices(texts)
        vectorization_layer.adapt(train_text)
        return vectorization_layer

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
    ):
        AUTO = tf.data.AUTOTUNE
        slices = (texts) if labels is None else (texts, labels)
        ds = tf.data.Dataset.from_tensor_slices(slices)

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
            # Create the training dataset
            train_df = df[df["fold"] != fold].sample(frac=1)
            train_text = train_df["text"].to_list()
            train_labels = train_df["label"].to_list()

            # Vectorize the text based on training data
            vectorizer = self.buildVectorizationLayer(train_text)

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
        else:  # Create test data set
            if vectorizer is not None:
                test_text = df["text"].to_list()
                # Vectorize the text based on training data
                #         vectorizer = buildVectorizationLayer(test_text) # Create the vectorization layer
                test_ds = self.buildDataset(
                    test_text,
                    None,
                    batch_size=self.params.batch_size,
                    shuffle=False,
                    drop_remainder=False,
                    repeat=False,
                    vectorizer=vectorizer,
                )
                return test_ds
