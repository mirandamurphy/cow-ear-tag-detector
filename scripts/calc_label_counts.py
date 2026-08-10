import pandas as pd
import logging

from utils.definitions import PROCESSED_LABELS_DIR, LOGS_DIR
from utils.constants import CLASS_LABEL

logger = logging.getLogger(__name__)

def calc_num_labels_per_file():
    label_counts = {}

    for file in PROCESSED_LABELS_DIR.glob("*.txt"):
        with open(file) as f:
            # Get first part of line and add count if it equals class label (i.e., '0')
            # An extra line.split() is added in case any files have trailing white spaces
            label_counts[file.stem] = sum (1 for line in f if line.split() and line.split()[0] == CLASS_LABEL)
    return label_counts

def save_label_counts_to_df():
    logger.info("Calculating label counts per image.")
    label_counts = calc_num_labels_per_file()
    return pd.DataFrame(list(label_counts.items()), columns=['file', 'labels'])

def save_df_to_csv(df):
    csv_path = LOGS_DIR / "label_counts.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved label counts to CSV (%s)", csv_path)


if __name__ == "__main__":
    label_counts_df = save_label_counts_to_df()
    print(label_counts_df)
