import mlflow
from mlflow.entities import experiment
import pandas as pd
import importlib
import logging
from utils.definitions import MLFLOW_TRACKING_URI, RAW_IMAGES_DIR, RAW_LABELS_DIR

# Step 1
remove_duplicates = importlib.import_module("01_remove_duplicates")
# Step 2
remove_blurry = importlib.import_module("02_remove_blurry")
# Step 3
calc_label_counts = importlib.import_module("03_calc_label_counts")

logging = logging.getLogger(__name__)



def run_pipeline():

    logging.info("Starting data preprocessing pipeline...")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("cow-ear-tag-detector")

    with mlflow.start_run(run_name="dataset-prep-v1"):

        mlflow.log_artifacts(
            str(RAW_IMAGES_DIR),
            artifact_path="dataset/raw"
        )




if __name__ == "__main__":
    run_pipeline()