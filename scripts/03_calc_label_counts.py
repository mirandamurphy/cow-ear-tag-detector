import pandas as pd

from utils.definitions import INTERIM_LABELS_DIR
from utils.constants import CLASS_LABEL

def calc_num_labels_per_file():
    label_counts = {}

    for file in INTERIM_LABELS_DIR.glob("*.txt"):
        with open(file) as f:
            # Get first part of line and add count if it equals class label (i.e., '0')
            # An extra line.split() is added in case any files have trailing white spaces
            label_counts[file.stem] = sum (1 for line in f if line.split() and line.split()[0] == CLASS_LABEL)
    return label_counts

def save_label_counts_to_df():
    label_counts = calc_num_labels_per_file()
    return pd.DataFrame(list(label_counts.items()), columns=['file', 'labels'])


if __name__ == "__main__":
    label_counts_df =stats = save_label_counts_to_df()
    print(label_counts_df)
