# LLM Text Detection

## Motivation

The aim is to develop a text classification application to identify essays generated by Language Model Models (LLMs). Educators are concerned that LLMs may enable plagiarism, hindering students' skill development by allowing them to submit essays that are not their own. The application intends to address this issue by detecting essays produced with the assistance of LLMs.

This project is inspired by the [kaggle competition](https://www.kaggle.com/competitions/llm-detect-ai-generated-text)

## Implementation

- Model : A simple and efficient 1D-CNN network architecture for text classification
- Data  : [Training data](https://www.kaggle.com/datasets/thedrcat/daigt-proper-train-dataset)
- Pipeline Tracking : [DVC](https://dvc.org/)
- Experiment Tracking : [mlflow](https://mlflow.org/)
