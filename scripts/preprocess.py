import mlflow
import logging

from utils.definitions import MLFLOW_TRACKING_URI, RAW_IMAGES_DIR, RAW_LABELS_DIR
from utils.logger_config import setup_logging

import step_01_remove_duplicates as remove_duplicates
import step_02_remove_blurry as remove_blurry
import step_03_split_dataset as split_dataset

logger = logging.getLogger(__name__)

def log_raw_dataset_metadata():
    num_images = sum(1 for file in RAW_IMAGES_DIR.iterdir() if file.is_file())
    num_labels = sum(1 for file in RAW_LABELS_DIR.iterdir() if file.is_file())
    return {"raw_images": num_images, "raw_labels": num_labels}

def run_pipeline():

    logger.info("Starting data preprocessing pipeline...")

    # Set up Mlflow experiment
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("cow-ear-tag-detector")

    with mlflow.start_run(run_name="dataset-pipline-v1"):

        # Log starting dataset metadata
        metadata = log_raw_dataset_metadata()
        mlflow.log_metrics(metadata)

        # Step 1: Remove duplicate/similar images
        with mlflow.start_run(run_name="remove_duplicates", nested=True):
            metadata = remove_duplicates.run()
            mlflow.log_metrics(metadata)

        with mlflow.start_run(run_name="remove_blurry", nested=True):
            metadata = remove_blurry.run()
            mlflow.log_metrics(metadata)

        with mlflow.start_run(run_name="split_dataset", nested=True):
            metadata = split_dataset.run()
            mlflow.log_metrics(metadata)

    logger.info("Data preprocessing pipeline has been completed.")


if __name__ == "__main__":
    setup_logging()
    run_pipeline()