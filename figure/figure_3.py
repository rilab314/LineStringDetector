import os
import cv2
import numpy as np
import config as cfg

METAINFO = [
    {'id': 0, 'name': 'ignore', 'color': (0, 0, 0)},
    {'id': 1, 'name': 'center_line', 'color': (77, 77, 255)},
    {'id': 2, 'name': 'u_turn_zone_line', 'color': (77, 178, 255)},
    {'id': 3, 'name': 'lane_line', 'color': (77, 255, 77)},
    {'id': 4, 'name': 'bus_only_lane', 'color': (255, 153, 77)},
    {'id': 5, 'name': 'edge_line', 'color': (255, 77, 77)},
    {'id': 6, 'name': 'path_change_restriction_line', 'color': (178, 77, 255)},
    {'id': 7, 'name': 'no_parking_stopping_line', 'color': (77, 255, 178)},
    {'id': 8, 'name': 'guiding_line', 'color': (255, 178, 77)},
    {'id': 9, 'name': 'stop_line', 'color': (77, 102, 255)},
    {'id': 10, 'name': 'safety_zone', 'color': (255, 77, 128)},
    {'id': 11, 'name': 'bicycle_lane', 'color': (128, 255, 77)},
]

TARGET_CLASSES = ['guiding_line', 'safety_zone', 'lane_line']
TARGET_COLORS = [
    np.array(next(meta['color'] for meta in METAINFO if meta['name'] == cls), dtype=np.uint8)
    for cls in TARGET_CLASSES
]

def filter_image_by_colors(image_path: str, target_colors: list) -> list:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = []
    for color in target_colors:
        mask = cv2.inRange(image_rgb, color, color)
        output = np.zeros_like(image_rgb)
        output[mask == 255] = color
        results.append(output)

    return results


def create_grid_image(gt_folder, pred_folder, output_folder, separator_width=5):
    os.makedirs(output_folder, exist_ok=True)
    image_names = sorted(os.listdir(gt_folder))

    for img_name in image_names:
        try:
            gt_path = os.path.join(gt_folder, img_name)
            pred_path = os.path.join(pred_folder, img_name)

            gt_imgs = filter_image_by_colors(gt_path, TARGET_COLORS)
            pred_imgs = filter_image_by_colors(pred_path, TARGET_COLORS)

            height = max(img.shape[0] for img in gt_imgs + pred_imgs)
            width = max(img.shape[1] for img in gt_imgs + pred_imgs)
            gt_imgs = [cv2.resize(img, (width, height)) for img in gt_imgs]
            pred_imgs = [cv2.resize(img, (width, height)) for img in pred_imgs]

            v_sep = 255 * np.ones((height, separator_width, 3), dtype=np.uint8)

            row1 = gt_imgs[0]
            for img in gt_imgs[1:]:
                row1 = np.hstack((row1, v_sep, img))

            row2 = pred_imgs[0]
            for img in pred_imgs[1:]:
                row2 = np.hstack((row2, v_sep, img))

            h_sep = 255 * np.ones((separator_width, row1.shape[1], 3), dtype=np.uint8)

            final_image = np.vstack((row1, h_sep, row2))

            final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
            out_path = os.path.join(output_folder, img_name)
            cv2.imwrite(out_path, final_image_bgr)
            print(f"Saved: {out_path}")

        except Exception as e:
            print(f"[ERROR] Failed to process {img_name}: {e}")


if __name__ == "__main__":
    data_path = cfg.DATA_PATH
    gt_dir = os.path.join(data_path, 'color_annotations', 'validation')
    pred_dir = os.path.join(data_path, 'prediction')
    output_dir = os.path.join(data_path, 'figure', 'figure_3')

    create_grid_image(gt_dir, pred_dir, output_dir)
