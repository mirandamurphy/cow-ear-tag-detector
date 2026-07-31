import mlflow
import importlib
import logging
from utils.definitions import MLFLOW_TRACKING_URI, RAW_IMAGES_DIR, RAW_LABELS_DIR

remove_duplicates = importlib.import_module("01_remove_duplicates")
remove_blurry = importlib.import_module("02_remove_blurry")
split_dataset = importlib.import_module("03_split_dataset")

logger = logging.getLogger(__name__)

def log_raw_dataset_stats():
    num_images = sum(1 for file in RAW_IMAGES_DIR.iterdir() if file.is_file())
    num_labels = sum(1 for file in RAW_LABELS_DIR.iterdir() if file.is_file())
    return {"raw_images": num_images, "raw_labels": num_labels}

def run_pipeline():

    logger.info("Starting data preprocessing pipeline...")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("cow-ear-tag-detector")

    with mlflow.start_run(run_name="dataset-pipline-v1"):
        with mlflow.start_run(run_name="remove_duplicates", nested=True):
            stats = remove_duplicates.remove_duplicates()
            mlflow.log_metrics(stats)

            with mlflow.start_run(run_name="remove_blurry", nested=True):
                stats = remove_blurry.remove_blurry_images()
                mlflow.log_metrics(stats)

            with mlflow.start_run(run_name="remove_blurry", nested=True):
                stats = remove_blurry.remove_blurry_images()
                mlflow.log_metrics(stats)

            with mlflow.start_run(run_name="split_dataset", nested=True):
                stats = split_dataset.split_dataset()
                mlflow.log_metrics(stats)

        logger.info("Pipeline complete.")



if __name__ == "__main__":
    run_pipeline()