import logging
import importlib
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from utils.definitions import INTERIM_IMAGES_DIR, INTERIM_LABELS_DIR, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, TEST_IMAGES_DIR, TEST_LABELS_DIR, VAL_IMAGES_DIR, VAL_LABELS_DIR

logger = logging.getLogger(__name__)

# Step 3
calc_label_counts = importlib.import_module("03_calc_label_counts")


label_counts_df = calc_label_counts.save_label_counts_to_df()
