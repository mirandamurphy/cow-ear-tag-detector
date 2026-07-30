import shutil
import logging
import imagehash
from PIL import Image
from utils.definitions import RAW_IMAGES_DIR, RAW_LABELS_DIR, INTERIM_IMAGES_DIR, INTERIM_LABELS_DIR
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
            logger.info("Keeping image: %s", path)
    return kept_images


def copy_unique_data(kept_images):

    image_stems = {path.stem for path in kept_images}
    label_stems = {path.stem for path in RAW_LABELS_DIR.glob("*.txt")}

    matched_stems = image_stems.intersection(label_stems)

    # Copy labels that have an image being kept
    for label_path in RAW_LABELS_DIR.glob("*.txt"):
        if label_path.stem in matched_stems:
            shutil.copy2(label_path, INTERIM_LABELS_DIR)
            logger.info("Moved label: %s", label_path.stem)

    # Copy images that have a corresponding label
    for image_path in kept_images:
        if image_path.stem in matched_stems:
            shutil.copy2(image_path, INTERIM_IMAGES_DIR)
            logger.info("Saved image %s", image_path.stem)
        else:
            logger.warning("MISSING LABEL FILE: image [ %s ] is not being kept.", image_path.stem)



def remove_duplicates():
    logger.info("Beginning step 1, 01_remove_duplicates.py...")
    p_hashes = compute_perceptual_hashes()
    unique_images = find_unique_images(p_hashes)
    copy_unique_data(unique_images)
    logger.info("01_remove_duplicates.py is complete")

    # mlflow Metadata
    return {
        "input_images": len(p_hashes),
        "remaining_images": len(unique_images),
        "duplicates_removed": len(p_hashes) - len(unique_images)
    }


if __name__ == "__main__":
    stats = remove_duplicates()
    print(stats)



