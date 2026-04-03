import os
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple, Dict
from pycocotools import mask as maskUtils
import config as cfg

# ================= Configuration =================
# 경로 설정 (사용자 환경에 맞게 수정 필요)
JSON_PATH = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Archive/Dataset/unzips/LaneDetector(copy)/test/instances_validation2017.json'  # 입력 JSON 경로
IMAGE_DIR = os.path.join(cfg.DATA_PATH, 'images', 'validation')  # 원본 이미지 폴더 (해상도 확인용)
SAVE_DIR = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Archive/Dataset/unzips/LaneDetector(copy)/test/result'  # 결과 저장 폴더
os.makedirs(SAVE_DIR, exist_ok=True)

# 병합 설정
CONNECTION_THRESHOLD = 10  # 픽셀 단위: 이 거리 이내에 있으면 연결 시도
SAMPLE_STRIDE = 15  # 샘플링 간격
THICKNESS = 3  # 마스크 그리기 두께

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


@dataclass
class LineString:
    id: int
    class_id: int
    points: np.ndarray = None  # (N, 2) [x, y]


class LaneJsonMerger:
    def __init__(self, json_path, img_dir, save_dir):
        self.json_path = json_path
        self.img_dir = img_dir
        self.save_dir = save_dir
        self.meta_map = {item['id']: item for item in METAINFO}

        os.makedirs(self.save_dir, exist_ok=True)

        # Load JSON
        print(f"Loading JSON from {json_path}...")
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        # Group annotations by image_id
        self.img_to_anns = {}
        annotations = self.data.get('annotations', self.data)
        if isinstance(annotations, list):
            for ann in annotations:
                img_id = ann['image_id']
                if img_id not in self.img_to_anns:
                    self.img_to_anns[img_id] = []
                self.img_to_anns[img_id].append(ann)

        self.stats = []  # CSV 통계 저장용

    def process(self):
        new_annotations = []

        print("Processing images...")
        for img_id, anns in tqdm(self.img_to_anns.items()):
            # 1. 이미지 정보 확인 (해상도)
            img_name = f"{img_id}.png"  # 혹은 jpg, json 내 정보 확인 필요
            img_path = os.path.join(self.img_dir, img_name)

            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                h, w = img.shape[:2]
            else:
                # 이미지가 없으면 annotation 정보를 기반으로 추정하거나 기본값 사용
                # 여기서는 json 내 segmentation 정보로 대략적 크기 유추 가능하나
                # 안전하게 기본값 혹은 에러 처리
                h, w = 1080, 1920  # Default fallback

            img_shape = (h, w)
            count_before = len(anns)

            # 2. 클래스별 분류
            class_groups = {}
            for ann in anns:
                cid = ann['category_id']
                if cid not in class_groups:
                    class_groups[cid] = []
                class_groups[cid].append(ann)

            merged_lines_in_image = []

            # 3. 클래스별 처리
            for cid, class_anns in class_groups.items():
                # A. Polygon -> LineString 변환 (Thinning & Sampling)
                line_strings = self._convert_anns_to_linestrings(class_anns, img_shape, cid)

                # B. 병합 (Merging Adjacent Lines)
                merged_lines = self._merge_lines_iterative(line_strings, img_shape)
                merged_lines_in_image.extend(merged_lines)

            count_after = len(merged_lines_in_image)
            self.stats.append({'image_id': img_id, 'before': count_before, 'after': count_after})

            # 4. 결과 축적 (RLE 인코딩)
            for line in merged_lines_in_image:
                new_ann = self._create_annotation(line, img_id, img_shape)
                new_annotations.append(new_ann)

        # 저장
        self._save_results(new_annotations)

    def _convert_anns_to_linestrings(self, anns, img_shape, class_id) -> List[LineString]:
        """JSON Polygon -> Binary Mask -> Thinning -> Sampled LineString"""
        line_strings = []
        uid = 0

        for ann in anns:
            # 1. Draw Polygon on Mask
            mask = np.zeros(img_shape, dtype=np.uint8)
            for seg in ann['segmentation']:
                pts = np.array(seg).reshape((-1, 1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [pts], 255)

            # 2. Thinning
            # 객체가 너무 작아 Thinning이 안될 수도 있으므로 예외처리 필요
            try:
                thinned = cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            except:
                thinned = mask  # fallback

            # 3. Sampling
            points = self._sample_points(thinned, SAMPLE_STRIDE)

            if points.shape[0] >= 2:
                line_strings.append(LineString(id=uid, class_id=class_id, points=points))
                uid += 1

        return line_strings

    def _merge_lines_iterative(self, line_strings: List[LineString], img_shape) -> List[LineString]:
        """반복적으로 가까운 선들을 병합"""
        changed = True
        while changed:
            changed = False
            merged_list = []
            used_ids = set()

            # 모든 쌍에 대해 거리 계산 (Brute-force approach for accuracy)
            # N이 작으므로 (보통 이미지당 객체 < 50개) 충분히 빠름
            for i in range(len(line_strings)):
                if line_strings[i].id in used_ids:
                    continue

                base_line = line_strings[i]
                best_match_idx = -1
                min_dist = CONNECTION_THRESHOLD

                # base_line의 양 끝점
                head1, tail1 = base_line.points[0], base_line.points[-1]

                for j in range(i + 1, len(line_strings)):
                    if line_strings[j].id in used_ids:
                        continue

                    target_line = line_strings[j]
                    head2, tail2 = target_line.points[0], target_line.points[-1]

                    # 4가지 경우의 수 거리 계산 (Head-Head, Head-Tail, Tail-Head, Tail-Tail)
                    # 실제로는 방향성이 있으므로 Tail-Head 혹은 Head-Tail 연결이 자연스러움
                    d1 = np.linalg.norm(head1 - head2)
                    d2 = np.linalg.norm(head1 - tail2)
                    d3 = np.linalg.norm(tail1 - head2)
                    d4 = np.linalg.norm(tail1 - tail2)

                    current_min = min(d1, d2, d3, d4)

                    if current_min < min_dist:
                        min_dist = current_min
                        best_match_idx = j

                if best_match_idx != -1:
                    # 병합 대상 발견
                    target_line = line_strings[best_match_idx]
                    merged_line = self._merge_two_lines(base_line, target_line, img_shape)
                    merged_list.append(merged_line)
                    used_ids.add(base_line.id)
                    used_ids.add(target_line.id)
                    changed = True
                else:
                    # 병합 대상 없음 -> 다음 루프를 위해 일단 보류하거나,
                    # 여기서 바로 처리하지 않고 남겨둠 (나중에 used_ids에 없는 것 추가)
                    pass

            # 병합되지 않은 나머지 선들 추가
            for line in line_strings:
                if line.id not in used_ids:
                    merged_list.append(line)

            line_strings = merged_list
            # ID 재정렬
            for idx, line in enumerate(line_strings):
                line.id = idx

        return line_strings

    def _merge_two_lines(self, line1: LineString, line2: LineString, img_shape) -> LineString:
        """두 LineString을 시각적으로 연결한 후 다시 Sampling하여 하나로 만듦"""
        h, w = img_shape
        canvas = np.zeros((h, w), dtype=np.uint8)

        # 두 선 그리기
        pts1 = line1.points.reshape((-1, 1, 2))
        pts2 = line2.points.reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts1], False, 255, 1)
        cv2.polylines(canvas, [pts2], False, 255, 1)

        # 가장 가까운 끝점끼리 직선 연결
        endpoints1 = [line1.points[0], line1.points[-1]]
        endpoints2 = [line2.points[0], line2.points[-1]]

        min_d = float('inf')
        connect_pair = (None, None)

        for p1 in endpoints1:
            for p2 in endpoints2:
                d = np.linalg.norm(p1 - p2)
                if d < min_d:
                    min_d = d
                    connect_pair = (p1, p2)

        if connect_pair[0] is not None:
            cv2.line(canvas, tuple(connect_pair[0]), tuple(connect_pair[1]), 255, 1)

        # 다시 Sampling (하나의 매끄러운 선으로 만들기 위함)
        # 1. Thinning (연결 부위가 두꺼워졌을 수 있으므로)
        thinned = cv2.ximgproc.thinning(canvas, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        # 2. Re-sampling
        new_points = self._sample_points(thinned, SAMPLE_STRIDE)

        return LineString(id=line1.id, class_id=line1.class_id, points=new_points)

    def _sample_points(self, line_img: np.ndarray, stride: int) -> np.ndarray:
        # lane_detector.py의 로직 차용
        rows, cols = np.nonzero(line_img)
        if len(rows) == 0: return np.array([])
        points = np.stack((cols, rows), axis=1)
        if points.shape[0] < 2:
            return points

        # 시작점 찾기 (y가 가장 큰 점 or x가 가장 작은 점 등 휴리스틱)
        # 여기서는 PCA나 끝점 탐색 대신 간단히 y좌표 최하단(이미지 아래쪽)을 시작점으로 가정
        # 혹은 lane_detector의 _sort_to_direction 로직을 완전 구현해야 함.
        # 여기서는 간단한 Nearest Neighbor Sort 구현

        sorted_indices = []
        current_idx = np.argmax(points[:, 1])  # 가장 아래쪽 점
        sorted_indices.append(current_idx)

        remaining_indices = set(range(len(points)))
        remaining_indices.remove(current_idx)

        while remaining_indices:
            last_pt = points[sorted_indices[-1]]
            # 남은 점들과의 거리 계산
            rem_list = list(remaining_indices)
            dists = np.sum((points[rem_list] - last_pt) ** 2, axis=1)
            nearest_in_rem = np.argmin(dists)
            nearest_idx = rem_list[nearest_in_rem]

            # 너무 멀리 튀면(노이즈) 종료 가능
            if dists[nearest_in_rem] > (stride * 5) ** 2:  # 임계값
                break

            sorted_indices.append(nearest_idx)
            remaining_indices.remove(nearest_idx)

        sorted_points = points[sorted_indices]

        # Stride 적용 (너무 촘촘한 점 제거)
        final_points = [sorted_points[0]]
        for pt in sorted_points[1:]:
            if np.linalg.norm(pt - final_points[-1]) >= stride:
                final_points.append(pt)

        return np.array(final_points).astype(np.int32)

    def _create_annotation(self, line: LineString, image_id: str, img_shape):
        h, w = img_shape
        # Draw line for RLE generation
        mask = np.zeros((h, w), dtype=np.uint8)
        if line.points is not None and len(line.points) > 1:
            pts = line.points.reshape((-1, 1, 2))
            cv2.polylines(mask, [pts], False, 1, thickness=THICKNESS)

        mask = np.asfortranarray(mask)
        rle = maskUtils.encode(mask)
        if isinstance(rle['counts'], bytes):
            rle['counts'] = rle['counts'].decode('utf-8')

        return {
            "image_id": image_id,
            "category_id": int(line.class_id),
            "segmentation": rle,
            "score": 1.0
        }

    def _save_results(self, annotations):
        # JSON 저장
        out_json_path = os.path.join(self.save_dir, 'merged_annotations.json')
        print(f"Saving merged JSON to {out_json_path}...")
        # 원본의 info, licenses 등을 유지하려면 self.data에서 복사 필요
        output_data = self.data.copy()
        output_data['annotations'] = annotations  # 덮어쓰기

        with open(out_json_path, 'w') as f:
            json.dump(output_data, f)

        # CSV 통계 저장
        csv_path = os.path.join(self.save_dir, 'merge_stats.csv')
        print(f"Saving Stats CSV to {csv_path}...")
        df = pd.DataFrame(self.stats)

        # 총계 계산 및 추가
        if not df.empty:
            total_before = df['before'].sum()
            total_after = df['after'].sum()
            total_row = pd.DataFrame([{'image_id': 'TOTAL', 'before': total_before, 'after': total_after}])
            df = pd.concat([df, total_row], ignore_index=True)

        df.to_csv(csv_path, index=False)
        print("All processes done.")


# ================= 실행 =================
if __name__ == "__main__":
    merger = LaneJsonMerger(JSON_PATH, IMAGE_DIR, SAVE_DIR)
    merger.process()