import shutil
import imagehash

from PIL import Image

import logging

from utils.definitions import RAW_IMAGES_DIR, RAW_LABELS_DIR, PROCESSED_IMAGES_DIR, PROCESSED_LABELS_DIR
from utils.constants import P_HASH_THRESHOLD

logger = logging.getLogger(__name__)

# Converts image to RBG to ensure consist color format and computes perceptual hash
def compute_perceptual_hashes():

    image_hashes = {}

    for path in RAW_IMAGES_DIR.glob("*.jpg"):
        if path.is_file():
            with Image.open(path) as img:
                logger.info("Hashing image : %s", path.name)
                image_hashes[path] = imagehash.phash(img.convert("RGB"))
    return image_hashes


def find_unique_images(image_hashes):

    kept_images = []
    kept_hashes = []

    # Compare image hashes to filter out duplicate/similar images
    for path, p_hash in image_hashes.items():
        if all(
                (p_hash - existing_hash) >= P_HASH_THRESHOLD
                for existing_hash in kept_hashes
        ):
            kept_images.append(path)
            kept_hashes.append(p_hash)
            logger.info("Keeping image: %s", path.name)
    return kept_images


def copy_unique_data(kept_images):

    image_stems = {path.stem for path in kept_images}
    label_stems = {path.stem for path in RAW_LABELS_DIR.glob("*.txt")}

    matched_stems = image_stems.intersection(label_stems)

    # Copy labels that have an image being kept
    for label_path in RAW_LABELS_DIR.glob("*.txt"):
        if label_path.stem in matched_stems:
            shutil.copy2(label_path, PROCESSED_LABELS_DIR)
            logger.info("Moved label: %s", label_path.name)

    copied_count = 0

    # Copy images that have a corresponding label
    for image_path in kept_images:
        if image_path.stem in matched_stems:
            shutil.copy2(image_path, PROCESSED_IMAGES_DIR)
            copied_count += 1
            logger.info("SAVED IMAGE: %s", image_path.name)
        else:
            logger.warning("MISSING LABEL FILE: image [ %s ] is not being kept.", image_path.name)
    return copied_count

def run():
    logger.info("BEGINNING STEP 1: REMOVE DUPLICATES...")
    p_hashes = compute_perceptual_hashes()
    unique_images = find_unique_images(p_hashes)
    copied_count = copy_unique_data(unique_images)



    logger.info(
        "COMPLETED STEP 1: REMOVE DUPLICATES. input_images: %d, unique_images: %d, remaining_images: %d, duplicates removed %d, images_missing_labels: %d",
        len(p_hashes), len(unique_images), copied_count, (len(p_hashes) - len(unique_images)), (len(unique_images) - copied_count))

    # mlflow Metadata
    return {
        "input_images": len(p_hashes),
        "unique_images": len(unique_images),
        "remaining_images": copied_count,
        "duplicates_removed": len(p_hashes) - len(unique_images),
        "images_missing_labels": len(unique_images) - copied_count
    }


if __name__ == "__main__":
    metadata = run()
    print(metadata)



