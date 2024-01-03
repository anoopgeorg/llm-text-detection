import os
import pandas as pd
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin

from src.llmTextDetection import logger, logflow
from src.llmTextDetection.pipeline.prediction_pipe import PredictionPipeLine
from src.llmTextDetection.config.configuration import configManager


application = Flask("__name__")
CORS(application)


class WebInstance:
    def __init__(self):
        config_manager = configManager()
        prediction_config = config_manager.getPredictionConfig()
        self.predictor = PredictionPipeLine(
            model_path=prediction_config.default_model,
            vectorizer_path=prediction_config.default_vectorizer,
        )
        logger.info("Web Instance has been instantiated")


@logflow
@application.route("/", methods=["GET", "POST"])
@cross_origin()
def home():
    logger.info("Home page rendered")
    result = None
    if request.method == "POST":
        essay_input = request.form.get("essayInput")
        data = {"text": [essay_input]}
        df = pd.DataFrame(data)
        prediction = web_app.predictor.makePrediction(test_df=df).flatten()
        logger.info(str(prediction[0]))
        if prediction[0] > 0.8:
            result = "AI"
        else:
            result = "Human"
    return render_template("index.html", prediction_result=result)


@logflow
@application.route("/train", methods=["GET"])
@cross_origin()
def train():
    os.system("dvc repro")
    logger.info("Train Evaluation Pipeline completed successfully!")
    return render_template("<h1>Training Completed Successfully!</h1>")


if __name__ == "__main__":
    web_app = WebInstance()
    application.run(host="0.0.0.0", port=8080)
