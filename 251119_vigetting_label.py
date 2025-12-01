import os
import cv2
import numpy as np
from tqdm import tqdm


RAW_DIR = r"D:\Left_DeepLearning\data\raw_for_dl"
LABEL_DIR = r"D:\Left_DeepLearning\data\label_for_dl"
NUM_IMAGES = 88
IMG_EXT = ".jpg"

def load_images(raw_dir, count):
    imgs = []
    for i in tqdm(range(count)):
        path = os.path.join(raw_dir, f"{i}{IMG_EXT}")
        img = cv2.imread(path)
    return imgs

def generate_flat_field(imgs):
    stack = np.array(imgs)
    median_img = np.median(stack, axis=0).astype(np.uint8)
    cv2.imwrite(os.path.join(LABEL_DIR, "debug_median.jpg"), median_img)
    flat_field = cv2.GaussianBlur(median_img, ksize=(251, 251), sigmaX=0)
    cv2.imwrite(os.path.join(LABEL_DIR, "debug_flat_field.jpg"), flat_field)
    return flat_field.astype(np.float32)

def generate_labels(imgs, flat_field):

    flat_mean = np.mean(flat_field, axis=(0, 1))
    if not os.path.exists(LABEL_DIR):
        os.makedirs(LABEL_DIR)

    for i, raw_img in enumerate(tqdm(imgs)):
        raw_float = raw_img.astype(np.float32)
        corrected = (raw_float / (flat_field + 1e-5)) * flat_mean
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        save_path = os.path.join(LABEL_DIR, f"{i}{IMG_EXT}")
        cv2.imwrite(save_path, corrected)


if __name__ == "__main__":
    if not os.path.exists(LABEL_DIR):
        os.makedirs(LABEL_DIR)
    images = load_images(RAW_DIR, NUM_IMAGES)

    if len(images) > 0:
        flat_field = generate_flat_field(images)
        generate_labels(images, flat_field)

        print("\n处理完成")
