import os
import numpy as np
import cv2

# METAINFO에 정의된 ID별 색상 매핑
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

# ID별 색상 매핑 딕셔너리 생성
COLOR_MAP = {info["id"]: info["color"] for info in METAINFO}


def filter_class_with_background(img, class_id):
    """
    특정 class_id에 해당하는 픽셀은 원래 색상 유지,
    나머지 픽셀은 20배 증가시켜 어두운 회색 계열로 유지.
    """
    h, w = img.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    color = COLOR_MAP[class_id]  # 해당 class의 색상 가져오기

    mask = (img == class_id)
    background = np.clip(img * 20, 0, 255).astype(np.uint8)  # 배경 강조 (0~255 범위 유지)
    color_img[:, :, 0] = background  # B
    color_img[:, :, 1] = background  # G
    color_img[:, :, 2] = background  # R
    color_img[mask] = color  # 해당 class_id 픽셀을 원래 색상으로 변경

    return color_img


def visualize_class_per_step(gt, pred, file_name):
    """
    GT 이미지와 예측 이미지를 한 번에 한 클래스씩 출력.
    """
    for info in METAINFO:
        class_id = info["id"]
        class_name = info["name"]

        gt_filtered = filter_class_with_background(gt, class_id)
        pred_filtered = filter_class_with_background(pred, class_id)

        gt_resized = cv2.resize(gt_filtered, (512, 512), interpolation=cv2.INTER_NEAREST)
        pred_resized = cv2.resize(pred_filtered, (512, 512), interpolation=cv2.INTER_NEAREST)

        comparison_img = cv2.hconcat([gt_resized, pred_resized])  # 좌우로 연결

        cv2.imshow(f"Class {class_id}: {class_name} (GT vs Pred)", comparison_img)
        print(f"[INFO] {file_name} - 현재 클래스: {class_id} ({class_name})")
        print("아무 키나 누르면 다음 클래스로 이동합니다.")

        cv2.waitKey(0)  # 키 입력 대기
        cv2.destroyAllWindows()


def process_images(gt_folder, pred_folder):
    """
    GT와 Pred가 둘 다 존재하는 이미지를 대상으로 클래스별로 하나씩 출력.
    """
    gt_files = set(f for f in os.listdir(gt_folder) if f.endswith('.png'))
    pred_files = set(f for f in os.listdir(pred_folder) if f.endswith('.png'))

    common_files = sorted(list(gt_files & pred_files))  # GT와 Pred 둘 다 존재하는 파일만 선택

    if not common_files:
        print("GT와 예측된 이미지가 모두 존재하는 파일이 없습니다.")
        return

    for file_name in common_files:
        gt_path = os.path.join(gt_folder, file_name)
        pred_path = os.path.join(pred_folder, file_name)

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        # ⚠️ GT 라벨링을 1~12 → 0~11로 변환
        gt = np.clip(gt - 1, 0, 11)

        print(f"[INFO] {file_name} 처리 중...")
        visualize_class_per_step(gt, pred, file_name)


# GT, 예측 이미지 폴더 경로 설정
gt_folder = "/home/humpback/youn_ws/LaneDetector_rilab/dataset/mask2former/annotation/validation"
pred_folder = "/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2025_LaneDetector/class_img/all_class_merged"

# 실행
process_images(gt_folder, pred_folder)
