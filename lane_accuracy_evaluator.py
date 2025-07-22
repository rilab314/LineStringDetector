import os
import numpy as np
import cv2
from collections import defaultdict

# 클래스 메타 정보
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


def visualize_images(origin_img, gt, pred, post_processing, file_name, show=False, save=False, save_dir=None):
    gt_color = convert_to_color(gt)
    pred_color = convert_to_color(pred)
    post_processing_color = convert_to_color(post_processing)

    sep_width = 5
    h, w = gt.shape
    separator = np.ones((h, sep_width, 3), dtype=np.uint8) * 255
    comparison_img = cv2.hconcat([origin_img, separator, gt_color, separator, pred_color, separator, post_processing_color])

    comparison_img = cv2.resize(comparison_img, (comparison_img.shape[1] // 2, comparison_img.shape[0] // 2))

    if save and save_dir:
        base_name = os.path.splitext(file_name)[0]
        save_path = os.path.join(save_dir, f"{base_name}.png")
        cv2.imwrite(save_path, comparison_img)
        print(f"[SAVE] 비교 이미지 저장됨: {save_path}")

    if show:
        cv2.imshow('Original | GT | Pred | Post-processing', comparison_img)
        print(f"[INFO] 현재 비교 중인 이미지: {file_name}")
        print("Enter 키를 누르면 다음 이미지로 진행합니다.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def evaluate_class_accuracy(data_path, show_images=True, save_images=True):
    origin_folder = os.path.join(data_path, 'images', 'validation')
    gt_folder = os.path.join(data_path, 'annotations', 'validation')
    pred_folder = os.path.join(data_path, 'pred_images')
    post_processing_folder = os.path.join(data_path, 'post_processing')


    origin_files = set(f for f in os.listdir(origin_folder) if f.endswith('.png'))
    gt_files = set(f for f in os.listdir(gt_folder) if f.endswith('.png'))
    pred_files = set(f for f in os.listdir(pred_folder) if f.endswith('.png'))
    post_processing_files = set(f for f in os.listdir(post_processing_folder) if f.endswith('.png'))

    common_files = sorted(list(gt_files & pred_files & post_processing_files))
    total_images = len(common_files)

    if not common_files:
        print("GT, Pred, Post-processing 모두 존재하는 파일이 없습니다.")
        return

    class_acc_pred = defaultdict(list)
    class_acc_post = defaultdict(list)

    save_dir = os.path.join(data_path, 'evaluate_images') if save_images else None
    if save_images:
        os.makedirs(save_dir, exist_ok=True)

    for idx, file_name in enumerate(common_files):
        origin_path = os.path.join(origin_folder, file_name)
        gt_path = os.path.join(gt_folder, file_name)
        pred_path = os.path.join(pred_folder, file_name)
        post_path = os.path.join(post_processing_folder, file_name)

        origin_img = cv2.imread(origin_path)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        ignore_mask = gt == 0.
        gt = gt - 1
        gt[ignore_mask] = 0

        pred_rgb = cv2.cvtColor(cv2.imread(pred_path), cv2.COLOR_BGR2RGB)
        pred = color2class_id(pred_rgb)

        post_rgb = cv2.cvtColor(cv2.imread(post_path), cv2.COLOR_BGR2RGB)
        post = color2class_id(post_rgb)

        visualize_images(origin_img, gt, pred, post, file_name, show=show_images, save=save_images, save_dir=save_dir)

        for class_id in range(1, 12):  # 클래스 0은 무시
            gt_mask = (gt == class_id)
            if np.sum(gt_mask) == 0:
                continue

            pred_mask = (pred == class_id)
            post_mask = (post == class_id)

            acc_pred = np.sum(gt_mask & pred_mask) / np.sum(gt_mask)
            acc_post = np.sum(gt_mask & post_mask) / np.sum(gt_mask)

            class_acc_pred[class_id].append(acc_pred)
            class_acc_post[class_id].append(acc_post)

        print(f"[INFO] {idx + 1}/{total_images} 처리 중: {file_name}")

    output_txt = os.path.join(data_path, 'accuracy.txt')
    with open(output_txt, 'w') as f:
        total_pred, total_post, count = 0, 0, 0

        f.write(f"{'Class':<32} | {'Pred':>10} | {'Post':>10} | {'Diff':>8}\n")
        f.write("-" * 64 + "\n")

        for class_id in sorted(class_acc_pred.keys()):
            avg_pred = np.mean(class_acc_pred[class_id])
            avg_post = np.mean(class_acc_post[class_id])
            diff = avg_post - avg_pred

            class_name = CLASS_NAMES[class_id]
            f.write(f"{class_name:<32} | {avg_pred:10.4f} | {avg_post:10.4f} | {diff:+8.4f}\n")

            print(f"{class_name:<32} | {avg_pred:.4f} | {avg_post:.4f} | {diff:+.4f}")

            total_pred += avg_pred
            total_post += avg_post
            count += 1

        mean_pred = total_pred / count if count > 0 else 0
        mean_post = total_post / count if count > 0 else 0
        mean_diff = mean_post - mean_pred

        f.write("-" * 64 + "\n")
        f.write(f"{'Mean':<32} | {mean_pred:10.4f} | {mean_post:10.4f} | {mean_diff:+8.4f}\n")

        print(f"\n[RESULT] Mean Accuracy Diff (Post - Pred): {mean_diff:+.4f}")
        print(f"[INFO] 결과가 저장되었습니다: {output_txt}")


def main():
    data_path = 'path'

    evaluate_class_accuracy(
        data_path=data_path,
        show_images=False,  # OpenCV 창 표시 여부
        save_images=True    # 이미지 저장 여부
    )


if __name__ == '__main__':
    main()
