import logging
import shutil

from sklearn.model_selection import train_test_split
from scripts import calc_label_counts

from utils.definitions import (
    PROCESSED_IMAGES_DIR,
    PROCESSED_LABELS_DIR,
    TRAIN_IMAGES_DIR,
    VAL_IMAGES_DIR,
    TEST_IMAGES_DIR,
    TRAIN_LABELS_DIR,
    VAL_LABELS_DIR,
    TEST_LABELS_DIR,
)

IMAGE_SPLIT_DIRS = {
    "train": TRAIN_IMAGES_DIR,
    "val": VAL_IMAGES_DIR,
    "test": TEST_IMAGES_DIR,
}

LABEL_SPLIT_DIRS = {
    "train": TRAIN_LABELS_DIR,
    "val": VAL_LABELS_DIR,
    "test": TEST_LABELS_DIR,
}

logger = logging.getLogger(__name__)

def move_split(df, split):

    image_output_dir = IMAGE_SPLIT_DIRS[split] # use key value (e.g., 'train')
    label_output_dir = LABEL_SPLIT_DIRS[split]

    # Make directory if it does not exist
    image_output_dir.mkdir(exist_ok=True)
    label_output_dir.mkdir(exist_ok=True)

    logger.info("Starting to move files for %s split...", split)

    # Using '_' because the DataFrame index is not needed
    for _, row in df.iterrows():
        image_input_path = PROCESSED_IMAGES_DIR / f"{row['file']}.jpg"
        label_input_path = PROCESSED_LABELS_DIR / f"{row['file']}.txt"

        if not image_input_path.exists():
            logger.warning("WARNING: Image %s not found, skipping.", image_input_path.stem)
            continue

        if not label_input_path.exists():
            logger.warning("WARNING: Label file %s not found, skipping.", label_input_path.stem)
            continue

        shutil.copy2(image_input_path, image_output_dir / image_input_path.name)
        logger.info("Image: %s copied to %s folder.", image_input_path.name, split)
        shutil.copy2(label_input_path, label_output_dir / label_input_path.name)
        logger.info("Label: %s copied to %s folder.", label_input_path.name, split)

    logging.info("File move is complete for %s split", split)



def run():


    dataset_df = calc_label_counts.save_label_counts_to_df()
    calc_label_counts.save_df_to_csv(dataset_df)

    logger.info("BEGINNING STEP 3: DATASET SPLIT...")

    # Split #1: 70% train and 30% test/val
    train_df, test_val_df = train_test_split(
        dataset_df,
        train_size=0.70,
        test_size=0.30, # val and test size combined (0.15 each)
        random_state=42, # randomization during splitting
    )

    # Split #2: Split remaining 30% (test/val) into 50% test, 50% val
    val_df, test_df = train_test_split(
        test_val_df,
        test_size=0.50,
        random_state=42,
    )

    move_split(train_df, "train")
    move_split(test_df, "test")
    move_split(val_df, "val")

    logger.info("COMPLETE STEP 3: DATASET SPLIT. Train size: %d, Val size: %d, Test size: %d",
                len(train_df), len(val_df), len(test_df))

    return {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
    }

if __name__ == "__main__":
   metadata = run()
   print(metadata)
