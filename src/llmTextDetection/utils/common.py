import os
import yaml
import json
import pickle
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any

from src.llmTextDetection import logger, logflow


@ensure_annotations
def read_yaml(path: Path) -> ConfigBox:
    """
        Reads YAML file and return ConfigBox object
    Args:
        path:Path - Path of YAML file
    Return:
        ConfigBox - ConfigBox object
    """
    try:
        with open(path, "r") as file:
            content = yaml.safe_load(file)
            logger.info(f"YAML file loadded successfully from: {path}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@logflow
def savePickle(obj, path):
    # Save the object using pickle.dump
    with open(path, "wb") as file:
        pickle.dump(obj, file)
        return True


@logflow
def loadPickle(path):
    # load the object using pickle.dump
    with open(path, "rb") as file:
        obj = pickle.load(file)
        return obj


@ensure_annotations
def create_directory(dir_list: list, verbose=True):
    """
        Creates the directories from a list
    Args:
        dir_list:list - list with paths of directories to be created
    """
    for path in dir_list:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """
        Saves JSON as a file
    Args:
        path: Path - Path for saving the file
        data: doct - Dictionary object of the JSON
    """
    with open(path, "w") as file:
        json.dump(data, file, indent=4)
    logger.info(f"JSON file saved successfully at: {path}")


def save_score(path: Path, data: dict):
    """
    Saves scores.JSON
    Args:
        path: Path - Path for saving the file
        data: dict - Dictionary object of the JSON
    """

    if path.is_file():
        # If the file exists, load its content
        with open(path, "r") as file:
            existing_data = json.load(file)

        # Update the existing data with the new data
        existing_data.update(data)
        data = existing_data
    else:
        path.touch(exist_ok=True)

    # Save the updated data to the file
    with open(path, "w") as file:
        json.dump(data, file, indent=4)

    logger.info(f"JSON file saved successfully at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
        Saves JSON as a file
    Args:
        path: Path - Path of JSON file
    """
    with open(path) as file:
        content = json.load(file)
    logger.info(f"JSON file loadded successfully from: {path}")
    return ConfigBox(content)


@logflow
@ensure_annotations
def find_latest_file(directory: Path) -> Path:
    # Ensure the path points to a directory
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a valid directory path.")

    # Get all files in the directory
    files = [file for file in directory.iterdir()]

    # If there are no files, return None
    if not files:
        return None

    # Find the latest file based on modification time
    latest_file = max(files, key=lambda file: file.stat().st_mtime)

    return latest_file
