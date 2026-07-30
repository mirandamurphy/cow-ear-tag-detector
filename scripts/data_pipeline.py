import mlflow
from mlflow.entities import experiment
import pandas as pd

from utils.definitions import MLFLOW_TRACKING_URI, RAW_IMAGES_DIR, RAW_LABELS_DIR





def run_pipeline():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("cow-ear-tag-detector")

    with mlflow.start_run(run_name="dataset-prep-v1"):

        mlflow.log_artifacts(
            str(RAW_IMAGES_DIR),
            artifact_path="dataset/raw"
        )


if __name__ == "__main__":
    run_pipeline()