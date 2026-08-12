from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Main directories
APP_DIR = ROOT_DIR / "app"
DATA_DIR = ROOT_DIR / "data"
DATASETS_DIR = ROOT_DIR / "datasets"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
SCRIPTS_DIR = ROOT_DIR / "scripts"


# MLflow (local)
MLFLOW_DB_PATH = ROOT_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
MLFLOW_ARTIFACTS_DIR = ROOT_DIR / "mlruns"

# Raw dataset
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_IMAGES_DIR = RAW_DATA_DIR / "images"
RAW_LABELS_DIR = RAW_DATA_DIR / "labels"

# Processed dataset
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_IMAGES_DIR = PROCESSED_DATA_DIR / "images"
PROCESSED_LABELS_DIR = PROCESSED_DATA_DIR / "labels"

# YOLO dataset
YOLO_DATASET_DIR = DATASETS_DIR / "yolo"
YOLO_IMAGES_DIR = YOLO_DATASET_DIR / "images"
YOLO_LABELS_DIR = YOLO_DATASET_DIR / "labels"

# YOLO splits
TRAIN_IMAGES_DIR = YOLO_IMAGES_DIR / "train"
VAL_IMAGES_DIR = YOLO_IMAGES_DIR / "val"
TEST_IMAGES_DIR = YOLO_IMAGES_DIR / "test"

TRAIN_LABELS_DIR = YOLO_LABELS_DIR / "train"
VAL_LABELS_DIR = YOLO_LABELS_DIR / "val"
TEST_LABELS_DIR = YOLO_LABELS_DIR / "test"

# YOLO config
DATA_YAML_PATH = YOLO_DATASET_DIR / "data.yaml"

# Models
BEST_MODEL_PATH = MODELS_DIR / "cow_ear_tag_detector_yolo26n.pt"

# Logs
PIPELINE_LOG_PATH = LOGS_DIR / "pipeline.log"


