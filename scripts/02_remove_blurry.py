import cv2
import numpy as np
import logging

from utils.definitions import INTERIM_IMAGES_DIR, INTERIM_LABELS_DIR
from utils.constants import FOCUS_SCORE_THRESHOLD

logger = logging.getLogger(__name__)

# Read each image in grayscale and pair with its 2D grayscale pixel array
def load_grayscale_images():

    grayscale_images = []
    for path in INTERIM_IMAGES_DIR.iterdir():
        if path.is_file():
            gray_image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) # cv2 expects string filename, not Path object
            if gray_image is not None:
                grayscale_images.append((path, gray_image))
                logger.info("Loaded grayscale image: %s", path.name)
            else:
                logger.warning("Failed to read image: %s", path.name)
    return grayscale_images


def compute_focus_score(gray_image: np.ndarray) -> float:
    """Processes image with Sobel operators in the x and y directions.
    Then calculates the mean of the squared gradient magnitudes across the image
    to combine the resulting gradients with the Tenegrad measure."""
    g_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    g_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    return (g_x**2 + g_y**2).mean() # Sharp images tend to have higher gradient values


# Filter out blurry images using the threshold value and remove blurry images from directory.
def remove_blurry_images():
    grayscale_images = load_grayscale_images()
    input_image_count = len(grayscale_images)
    removed_image_count = 0


    for image_path, gray_image in grayscale_images:
        focus_score = compute_focus_score(gray_image)
        blurry = focus_score < FOCUS_SCORE_THRESHOLD
        logger.info("Focus score for %s: %.2f (blurry=%s)", image_path.name, focus_score, blurry)

        if blurry:
            label_path = INTERIM_LABELS_DIR / f"{image_path.name}.txt"
            image_path.unlink() # Remove blurry images from working dataset
            logger.info("Removed blurry image: %s", image_path.name)
            removed_image_count += 1
            if label_path.exists():
                label_path.unlink()
                logger.info("Removed corresponding label: %s", label_path.name)
            else:
                logger.warning("No matching label file for: %s", image_path.stem)

    remaining_images_count = input_image_count - removed_image_count

    logger.info(
        "02_blur_filter.py is is complete. input: %d, removed: %d, remaining: %d",
        input_image_count, removed_image_count, remaining_images_count
    )

    return {
        "input_images" : input_image_count,
        "remaining_images": remaining_images_count,
        "blurry_images_removed": removed_image_count

    }

if __name__ == "__main__":
    stats = remove_blurry_images()
    print(stats)

