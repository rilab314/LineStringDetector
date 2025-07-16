import os
import numpy as np
import cv2
from collections import defaultdict

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

COLOR_MAP = {info['id']: info['color'] for info in METAINFO}
CLASS_NAMES = {info['id']: info['name'] for info in METAINFO}


def color2class_id(color_img):
    h, w, _ = color_img.shape
    class_map = np.zeros((h, w), dtype=np.uint8)
    for class_id, color in COLOR_MAP.items():
        mask = np.all(color_img == color, axis=-1)
        class_map[mask] = class_id
    return class_map


def convert_to_color(img):
    h, w = img.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in COLOR_MAP.items():
        mask = img == class_id
        color_img[mask] = color
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    return color_img


def visualize_images(gt, inference, pred, origin_img, file_name, save_dir=None):
    gt_color = convert_to_color(gt)
    inference_color = convert_to_color(inference)
    pred_color = convert_to_color(pred)
    # Add vertical white separators between images
    sep_width = 5
    h, w = gt.shape
    separator = np.ones((h, sep_width, 3), dtype=np.uint8) * 255
    comparison_img = cv2.hconcat([origin_img, separator, gt_color, separator, inference_color, separator, pred_color])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(file_name)[0]
        save_path = os.path.join(save_dir, f"{base_name}.png")
        cv2.imwrite(save_path, comparison_img)
        print(f"[SAVE] 비교 이미지 저장됨: {save_path}")
    else:
        cv2.imshow('GT | Inference | Pred', comparison_img)
        print(f"[INFO] 현재 비교 중인 이미지: {file_name}")
        print("Enter 키를 누르면 다음 이미지로 진행합니다.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def evaluate_class_accuracy(gt_folder, inference_folder, pred_folder, origin_folder, output_txt, show_images=True, save_dir=None):
    gt_files = set(f for f in os.listdir(gt_folder) if f.endswith('.png'))
    pred_files = set(f for f in os.listdir(pred_folder) if f.endswith('.png'))
    inference_files = set(f for f in os.listdir(inference_folder) if f.endswith('.png'))
    origin_files = set(f for f in os.listdir(origin_folder) if f.endswith('.png'))

    common_files = sorted(list(gt_files & pred_files & inference_files))
    total_images = len(common_files)

    if not common_files:
        print("GT, Inference, Pred 모두 존재하는 파일이 없습니다.")
        return

    class_acc_pred = defaultdict(list)
    class_acc_inf = defaultdict(list)

    for idx, file_name in enumerate(common_files):
        gt_path = os.path.join(gt_folder, file_name)
        inference_path = os.path.join(inference_folder, file_name)
        pred_path = os.path.join(pred_folder, file_name)
        origin_path = os.path.join(origin_folder, file_name)

        origin_img = cv2.imread(origin_path)

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt = np.clip(gt - 1, 0, 11)

        inference_bgr = cv2.imread(inference_path)
        inference_rgb = cv2.cvtColor(inference_bgr, cv2.COLOR_BGR2RGB)
        inference = color2class_id(inference_rgb)

        pred_bgr = cv2.imread(pred_path)
        pred_rgb = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB)
        pred = color2class_id(pred_rgb)

        if show_images or save_dir:
            visualize_images(gt, inference, pred, origin_img, file_name, save_dir)

        for class_id in range(1, 12):  # ID 0 = ignore
            # AFTER (정확한 비교를 위한 분리된 mask 정의)
            mask_pred = (gt == class_id) | (pred == class_id)
            mask_inf = (gt == class_id) | (inference == class_id)

            if np.sum(mask_inf) == 0:
                continue
            if np.sum(mask_pred) == 0:
                continue

            acc_pred = np.sum((gt == pred) & (gt == class_id)) / np.sum(mask_pred)
            class_acc_pred[class_id].append(acc_pred)

            acc_inf = np.sum((gt == inference) & (gt == class_id)) / np.sum(mask_inf)
            class_acc_inf[class_id].append(acc_inf)

        progress = (idx + 1) / total_images * 100
        print(f"[INFO] {idx + 1}/{total_images} ({progress:.2f}%) - {file_name} 처리 중...")

    with open(output_txt, 'w') as f:
        total_pred, total_inf, count = 0, 0, 0

        f.write(f"{'Class':<20} | {'Inference':>9} | {'Pred':>7} | {'Diff':>7}\n")
        f.write("-" * 55 + "\n")

        for class_id in sorted(class_acc_pred.keys()):
            avg_pred = np.mean(class_acc_pred[class_id])
            avg_inf = np.mean(class_acc_inf[class_id])
            delta = avg_pred - avg_inf

            class_name = CLASS_NAMES[class_id]
            f.write(f"{class_name:<20} | {avg_inf:9.4f} | {avg_pred:7.4f} | {delta:+7.4f}\n")

            print(f"{class_name:<20} | {avg_inf:.4f} | {avg_pred:.4f} | {delta:+.4f}")

            total_pred += avg_pred
            total_inf += avg_inf
            count += 1

        mean_pred = total_pred / count if count > 0 else 0
        mean_inf = total_inf / count if count > 0 else 0
        mean_diff = mean_pred - mean_inf

        f.write("-" * 55 + "\n")
        f.write(f"{'Mean':<20} | {mean_inf:9.4f} | {mean_pred:7.4f} | {mean_diff:+7.4f}\n")

        print(f"\n[RESULT] Mean Accuracy Diff (pred - inference): {mean_diff:+.4f}")
        print(f"[INFO] 결과가 저장되었습니다: {output_txt}")


def main():
    gt_folder = "/home/humpback/youn_ws/LaneDetector_rilab/dataset/mask2former/annotation/validation"
    inference_folder = "/home/humpback/youn_ws/LaneDetector_rilab/dataset/mask2former/pred_images"
    # pred_folder = "/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2025_LaneDetector/post_processing_inference"
    origin_folder = "/home/humpback/youn_ws/LaneDetector_rilab/dataset/mask2former/images"

    # pred_folder = "/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2025_LaneDetector/accuracy_img/250707_neighbor_lane"
    pred_folder = "/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2025_LaneDetector/accuracy_img/250707_only_findoverlap"

    output_txt = "/home/humpback/youn_ws/LaneDetector_rilab/dataset/mask2former/accuracy/accuracy_results.txt"

    evaluate_class_accuracy(
        gt_folder=gt_folder,
        inference_folder=inference_folder,
        pred_folder=pred_folder,
        origin_folder=origin_folder,
        output_txt=output_txt,
        show_images=False,
        save_dir='/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2025_LaneDetector/result/compare_img'
    )


if __name__ == '__main__':
    main()
