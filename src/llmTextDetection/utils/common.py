import os
import yaml
import json
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any

from src.llmTextDetection import logger


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
