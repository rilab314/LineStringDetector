import os
import cv2
import numpy as np
import config as cfg


def merge_image_triplets(image_folder: str, pred_folder: str, proc_folder: str, output_folder: str, separator_width: int = 5):
    os.makedirs(output_folder, exist_ok=True)
    image_names = sorted(os.listdir(image_folder))

    for img_name in image_names:
        path_a = os.path.join(image_folder, img_name)
        path_b = os.path.join(pred_folder, img_name)
        path_c = os.path.join(proc_folder, img_name)

        img_a = cv2.imread(path_a)
        img_b = cv2.imread(path_b)
        img_c = cv2.imread(path_c)

        if img_a is None or img_b is None or img_c is None:
            print(f"[WARN] One or more images missing or unreadable for {img_name}")
            continue

        height = max(img_a.shape[0], img_b.shape[0], img_c.shape[0])
        width = max(img_a.shape[1], img_b.shape[1], img_c.shape[1])
        img_a = cv2.resize(img_a, (width, height))
        img_b = cv2.resize(img_b, (width, height))
        img_c = cv2.resize(img_c, (width, height))

        separator = 255 * np.ones((height, separator_width, 3), dtype=np.uint8)
        combined = np.hstack((img_a, separator, img_b, separator, img_c))

        out_path = os.path.join(output_folder, img_name)
        cv2.imwrite(out_path, combined)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    data_path = cfg.DATA_PATH
    image_dir = os.path.join(data_path, 'images', 'validation')
    pred_dir = os.path.join(data_path, 'prediction')
    proc_dir = os.path.join(data_path, 'process', 'merged_lines')
    out_dir = os.path.join(data_path, 'figure', 'figure_1')

    merge_image_triplets(image_dir, pred_dir, proc_dir, out_dir)
