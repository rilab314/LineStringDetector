import os
import glob

import cv2
import numpy as np
import json
from pycocotools import mask as maskUtils
from typing import List, Tuple, Set, Dict
from dataclasses import dataclass
from tqdm import tqdm

from show_imgs import ImageShow
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


@dataclass
class LineString:
    id: int
    peak: Tuple[int, int]
    class_id: int
    points: np.ndarray = None  # 원본 선상의 샘플링된 점들 (N,2)
    ext_points: np.ndarray = None  # 양쪽으로 확장된 선상의 점들 ((N+M),2)
    src_range: Tuple[int, int] = None  # ext_points 내에서 원래 points가 차지하는 인덱스 범위
    length: float = 0  # 선의 길이 (유클리드 누적거리)


class LineStringDetector:
    id_offset = 10  # peak ID의 최소 오프셋
    sample_stride = 15  # 샘플링 간격 (픽셀)
    extend_len = 20  # 선 확장 길이 (픽셀)
    overlap_thresh = 2  # 겹치는 픽셀 수
    thickness = 5
    short_length = 30

    def __init__(self, data_path: str):
        self._data_path = data_path
        self._img_shape = (100, 100)
        self._palette = [info['color'][::-1] for info in METAINFO]
        print('palette:', self._palette)
        self._id_count = 0
        self._imshow_base = ImageShow('base images', columns=3, scale=0.8, enabled=True)
        self._imshow_proc = ImageShow('processing images', columns=3, scale=0.8, enabled=True)
        self._imshow_save = ImageShow('save images', columns=3, scale=1, enabled=True)
        self._exclude_classes = [0]

    def detect_line_strings(self):
        file_list = glob.glob(os.path.join(self._data_path, 'images', 'validation', '*.png'))
        file_list.sort()
        origin_json = []
        origin_excepted_json = []
        pred_json = []
        pred_excepted_json = []

        for i, file_name in enumerate(tqdm(file_list)):
            if i == 100:
                break
            print(f'===== [file_name] ===== {i} / {len(file_list)}, file:{file_name}')
            image, pred_img, anno_img = self._read_image(file_name)
            self._img_shape = image.shape[:2]
            self._id_count = self.id_offset

            origin_line_strings, line_img_raw = self.extract_lines(pred_img, file_name)
            origin_line_strings_excepted, line_img_raw_excepted = self._except_short_lines(origin_line_strings)
            line_strings, line_img_merged = self.merge_lines(origin_line_strings, 0)
            line_strings, line_img_merged = self.merge_lines(line_strings, 1)
            line_strings_excepted, line_img_excepted = self._except_short_lines(line_strings)

            images_to_save = {'src_img': image, 'anno_img': anno_img, 'pred_img': pred_img,
                              'raw_line': line_img_raw, 'merged_line': line_img_merged, 'long_line': line_img_excepted}
            self.save_images(images_to_save, file_name)
            image_id = os.path.basename(file_name).replace('.png', '')
            origin_json = self.accumulate_preds(origin_line_strings, image_id, origin_json)
            origin_excepted_json = self.accumulate_preds(origin_line_strings_excepted, image_id, origin_excepted_json)
            pred_json = self.accumulate_preds(line_strings, image_id, pred_json)
            pred_excepted_json = self.accumulate_preds(line_strings_excepted, image_id, pred_excepted_json)
            self._imshow_proc.display(1)

        with open(os.path.join(self._data_path, 'result', 'coco_pred_instances_origin.json'), 'w') as f:
            json.dump(origin_json, f)
        with open(os.path.join(self._data_path, 'result', 'coco_pred_instances_origin_excepted.json'), 'w') as f:
            json.dump(origin_excepted_json, f)
        with open(os.path.join(self._data_path, 'result', 'coco_pred_instances_merged.json'), 'w') as f:
            json.dump(pred_json, f)
        with open(os.path.join(self._data_path, 'result', 'coco_pred_instances_excepted.json'), 'w') as f:
            json.dump(pred_excepted_json, f)

    def _read_image(self, img_file: str):
        image = cv2.imread(img_file)
        print('image file', img_file)
        pred_file = img_file.replace('/images/validation', '/prediction/')
        pred_img = cv2.imread(pred_file)
        anno_file = img_file.replace('/images/', '/color_annotations/')
        anno_img = cv2.imread(anno_file)
        images = {'image': image, 'GT_img': anno_img, 'pred_img': pred_img}
        self._imshow_base.show_imgs(images)
        return image, pred_img, anno_img
    
    def extract_lines(self, pred_img: np.ndarray, file_name=None) -> Tuple[List[LineString], np.ndarray]:
        line_string_list = []
        for class_id, color in enumerate(self._palette):
            if class_id in self._exclude_classes:
                continue
            # for class_id in [1, 2, 4, 5, 7, 8, 9]:
            pred_class_map = np.all(pred_img == color, axis=-1).astype(np.uint8)
            line_map, line_strings = self._thin_image(pred_class_map, class_id)
            ext_lines = self._extend_lines(line_map, line_strings)
            line_string_list.extend(ext_lines)
        
        line_img = np.zeros_like(pred_img)
        line_img = self._draw_colored_lines(line_img, line_string_list, extended=True)
        self._imshow_proc.show(line_img, 'extracted lines')

        return line_string_list, line_img
    
    def merge_lines(self, src_line_strings: List[LineString], iter: int) -> Tuple[List[LineString], np.ndarray]:
        dst_line_strings = []
        print(f'=========== [merge_lines] iter={iter}, src_line_strings: {len(src_line_strings)}')
        for class_id, color in enumerate(self._palette):
            if class_id in self._exclude_classes:
                continue
            class_line_strings = [line for line in src_line_strings if line.class_id == class_id]
            merged_lines = self._merge_lines_by_class(class_line_strings, iter_count=1)
            print(f'[merge_lines] class_id={class_id}, src lines={len(class_line_strings)}, merged={len(merged_lines)}')
            dst_line_strings.extend(merged_lines)

        line_img = np.zeros([self._img_shape[0], self._img_shape[1], 3], dtype=np.uint8)
        line_img = self._draw_colored_lines(line_img, dst_line_strings)
        self._imshow_proc.show(line_img, f'merged_lines_{iter}')

        return dst_line_strings, line_img

    def _thin_image(self, seg_map: np.ndarray, class_id: int):
        print(f'----- [thin_image] -----')
        line_strings = []
        line_map = np.zeros_like(seg_map, dtype=np.int32)
        line_blobs = np.zeros_like(seg_map, dtype=np.int32)
        y, x = np.nonzero(seg_map)
        fill_value = self.id_offset

        show_blobs = line_blobs.copy()

        for k, (y, x) in enumerate(zip(y, x)):
            if line_blobs[y, x] > 0:
                continue

            # floodFill를 통해 seed가 포함된 blob을 채움
            temp = seg_map.copy()
            mask = np.zeros((seg_map.shape[0] + 2, seg_map.shape[1] + 2), np.uint8)
            cv2.floodFill(temp, mask, (x, y), fill_value)
            # 채워진 영역을 바이너리 마스크로 변환 (0 또는 255)
            line_blobs[temp == fill_value] = fill_value
            blob_mask = (temp == fill_value).astype(np.uint8) * 255

            show_blobs = line_blobs.astype(np.int16)

            # cv2.ximgproc.thinning 적용 (얇은 선 추출)
            # (cv2.ximgproc.thinning은 입력이 binary 이미지여야 함)
            line_img = cv2.ximgproc.thinning(blob_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

            # 결과를 line_map에 누적 (겹치는 영역은 덮어쓰기)
            line_map[line_img > 0] = fill_value
            line_strings.append(LineString(id=fill_value, class_id=class_id, peak=(x, y)))
            fill_value += 1

        return line_map.astype(np.uint8), line_strings

    def _extend_lines(self, line_map: np.ndarray, line_strings: List[LineString]) -> List[LineString]:
        print(f'----- [extend_lines] -----')
        id_list = np.unique(line_map)
        id_list = id_list[id_list >= self.id_offset]
        for line_string in line_strings:
            # 해당 라벨만 추출한 바이너리 이미지
            line_img = (line_map == line_string.id).astype(np.uint8)
            line_string.points = self._sample_points(line_img, self.sample_stride)
            if line_string.points.shape[0] < 2:
                line_string.id = None
                continue
            line_string.length = np.sum(np.linalg.norm(np.diff(line_string.points, axis=0), axis=1))
            if line_string.length < 3:
                line_string.id = None
                continue
            line_string = self._extrapolate_line(line_string, self.extend_len, self.sample_stride)

        line_strings = [ls for ls in line_strings if ls.id is not None]
        # 선 길이에 따라 내림차순 정렬
        line_strings.sort(key=lambda ls: ls.length, reverse=True)
        return line_strings

    def _sample_points(self, line_img: np.ndarray, stride: int) -> np.ndarray:
        rows, cols = np.nonzero(line_img)
        points = np.stack((cols, rows), axis=1)
        if points.shape[0] < 2:
            return points
        sorted_points = [points[0]]
        direction = points[1] - points[0]
        sorted_points = self._sort_to_direction(points, sorted_points, True, direction, stride)
        sorted_points = self._sort_to_direction(points, sorted_points, False, -direction, stride)
        sorted_points = np.array(sorted_points).astype(np.int32)
        return sorted_points.astype(np.int32)

    def _sort_to_direction(self, src_points: np.ndarray, sorted_points: List[np.ndarray], to_tail: bool,
                           direction: np.ndarray, stride: int) -> List[np.ndarray]:
        points = src_points.copy()
        while len(points) > 0:
            last_point = sorted_points[-1] if to_tail else sorted_points[0]
            distances = np.sqrt(np.sum((points - last_point) ** 2, axis=1))
            dot_products = np.sum((points - last_point) * direction, axis=1)
            valid_mask = (distances < 30) & (distances >= stride) & (dot_products >= 0)
            if np.sum(valid_mask) == 0:
                break
            distances[~valid_mask] = np.inf
            next_index = np.argmin(distances)
            if to_tail:
                sorted_points.append(points[next_index])
                # print(f'[sort_to_direction] next point: {sorted_points[-1]}')
                direction = sorted_points[-1] - last_point
            else:
                sorted_points.insert(0, points[next_index])
                # print(f'[sort_to_direction] next point: {sorted_points[0]}')
                direction = sorted_points[0] - last_point
            distances = np.sqrt(np.sum((points - last_point) ** 2, axis=1))
            points = points[distances >= stride]
        return sorted_points

    def _extrapolate_line(self, line_string: LineString, extend_len: int, stride: int) -> LineString:
        points = line_string.points  # (N,2) 배열
        N = len(points)
        head_prev_idx = min(max(N // 3, 3), N - 1)
        head_dir = points[0] - points[head_prev_idx]
        head_dir = head_dir / np.linalg.norm(head_dir)
        tail_prev_idx = N - 1 - min(max(N // 3, 3), N - 1)
        tail_dir = points[-1] - points[tail_prev_idx]
        tail_dir = tail_dir / np.linalg.norm(tail_dir)
        n_head = extend_len // stride
        head_ext = np.array([
            points[0] + head_dir * stride * i
            for i in range(1, n_head + 1)
        ])
        n_tail = extend_len // stride
        tail_ext = np.array([
            points[-1] + tail_dir * stride * i
            for i in range(1, n_tail + 1)
        ])
        ext_points = np.vstack([head_ext[::-1], points, tail_ext])
        ext_points = np.clip(ext_points, [0, 0], [self._img_shape[1], self._img_shape[0]])

        line_string.ext_points = np.rint(ext_points).astype(np.int32)
        line_string.src_range = (n_head, n_head + len(points) - 1)
        return line_string

    def _merge_lines_by_class(self, line_strings: List[LineString], iter_count: int = 0) -> List[LineString]:
        if len(line_strings) == 0:
            return []
        for line in line_strings:
            overlap_ids = self._find_overlap(line_strings, line)
            if len(overlap_ids) == 0:
                continue
            for overlap_id in overlap_ids:
                oppo_line = next((l for l in line_strings if l.id == overlap_id), None)
                if oppo_line is None:
                    continue
                if oppo_line.points is None:
                    continue
                combined_img = self._connect_tail2head(line, oppo_line)
                combined_img = cv2.cvtColor(combined_img, cv2.COLOR_BGR2GRAY)
                line.points = self._sample_points(combined_img, self.sample_stride)
                line.length = np.sum(np.linalg.norm(np.diff(line.points, axis=0), axis=1))
                line = self._extrapolate_line(line, self.extend_len, self.sample_stride)
                print(f'line {line.id} is merged with line {oppo_line.id}')
                oppo_line.id = None
                oppo_line.ext_points = None

        # id가 None이 아닌 선들만 남김
        line_strings = [line for line in line_strings if line.id is not None]
        return line_strings

    def _find_overlap(self, line_strings: List[LineString], this_line: LineString) -> Set[int]:
        src_line_id = this_line.id
        if this_line.ext_points is None:
            return set()
        line_strings_copy = line_strings.copy()
        line_strings_copy.remove(this_line)
        dilated_map = self._draw_line_strings(line_strings_copy, extend=True)
        dilated_map = cv2.dilate(dilated_map, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8))
        label_map = dilated_map.copy() if dilated_map.ndim == 2 else dilated_map[:, :, 0].copy()

        line_img = np.zeros_like(dilated_map, dtype=np.uint8)
        pts = this_line.ext_points.reshape((-1, 1, 2))
        cv2.polylines(line_img, [pts], isClosed=False, color=(255, 255, 255), thickness=3)
        line_map = line_img[:, :, 0]

        label_map[line_map == 0] = 0
        overlap_labels = label_map[label_map > 0]
        if overlap_labels.size == 0:
            return set()
        unique, counts = np.unique(overlap_labels, return_counts=True)

        mask_cond = (unique != 0) & (unique != src_line_id) & (counts > self.overlap_thresh)
        label_ids = set(unique[mask_cond].tolist())
        print(f'unique: {unique}, counts: {counts}, label_ids: {label_ids}')
        return label_ids

    def _connect_tail2head(self, line, oppo_line):
        image = np.zeros((self._img_shape[1], self._img_shape[0], 3), dtype=np.uint8)
        color = (line.id, line.id, line.id)
        line_pts = line.points.reshape((-1, 1, 2))
        oppo_pts = oppo_line.points.reshape((-1, 1, 2))
        cv2.polylines(image, [line_pts], isClosed=False, color=color, thickness=1)
        cv2.polylines(image, [oppo_pts], isClosed=False, color=color, thickness=1)

        this_endpoints = np.array([line.points[0], line.points[-1]])
        oppo_endpoints = np.array([oppo_line.points[0], oppo_line.points[-1]])
        distances = np.linalg.norm(this_endpoints[:, np.newaxis, :] - oppo_endpoints, axis=2)
        min_idx_flat = np.argmin(distances)
        this_idx, oppo_idx = np.unravel_index(min_idx_flat, distances.shape)
        print(f'min distance: {distances[this_idx, oppo_idx]:1.1f}, class_id: {line.class_id}, this_idx: {line.id}, oppo_idx: {oppo_line.id}')
        endpoints = [this_endpoints[this_idx], oppo_endpoints[oppo_idx]]
        cv2.line(image, endpoints[0], endpoints[1], color=color, thickness=1)
        return image

    def _except_short_lines(self, line_strings: List[LineString]) -> Tuple[List[LineString], np.ndarray]:
        filtered_line_strings = []
        for line in line_strings:
            line.length = np.sum(np.linalg.norm(np.diff(line.points, axis=0), axis=1))
            if line.length > self.short_length:
                filtered_line_strings.append(line)
        line_img = np.zeros([self._img_shape[0], self._img_shape[1], 3], dtype=np.uint8)
        line_img = self._draw_colored_lines(line_img, filtered_line_strings)
        return filtered_line_strings, line_img

    def _draw_line_strings(self, line_strings: List[LineString], extend=False, color=None):
        image = np.zeros((self._img_shape[0], self._img_shape[1], 3), dtype=np.uint8)
        for line in line_strings:
            if line.id is None or line.points is None:
                continue
            if extend:
                pts = line.ext_points.reshape((-1, 1, 2))
            else:
                pts = line.points.reshape((-1, 1, 2))
            line_color = (line.id, line.id, line.id) if color is None else color
            cv2.polylines(image, [pts], isClosed=False, color=line_color, thickness=3)
        return image

    def _draw_single_line(self, line_string : LineString, thickness=None, extend=False):
        image = np.zeros((self._img_shape[0], self._img_shape[1], 3), dtype=np.uint8)
        if line_string.id is None or line_string.points is None:
            return image
        pts = line_string.points.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=False, color=(line_string.id, line_string.id, line_string.id), thickness=thickness)
        return image

    def _draw_colored_lines(self, pred_img, line_strings : List[LineString], extended=False, select=None):
        image = pred_img.copy()
        for line in line_strings:
            if line.id is None:
                continue
            color = self._palette[line.class_id]
            if extended:
                pts = line.ext_points.reshape((-1, 1, 2))
            else:
                pts = line.points.reshape((-1, 1, 2))
            if select is None:
                cv2.polylines(image, [pts], isClosed=False, color=color, thickness=3)
            elif select is not None:
                if line.class_id == select:
                    cv2.polylines(image, [pts], isClosed=False, color=color, thickness=3)
                else:
                    continue
        return image

    def _draw_blobs_with_color(self, line_map):
        n_labels = int(np.max(line_map))
        rng = np.random.default_rng(42)  # 재현성을 위해 시드 고정(원하면 제거)
        H = rng.uniform(0, 180, size=n_labels + 1)  # [0,180)
        S = rng.uniform(170, 255, size=n_labels + 1)  # 채도 ↑ (170~255)
        V = rng.uniform(130, 220, size=n_labels + 1)  # 명도 ↓ (130~220)
        H[0], S[0], V[0] = 0, 0, 0
        hsv = np.stack([H, S, V], axis=1).astype(np.uint8).reshape(-1, 1, 3)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)  # shape: (n_labels+1, 3)
        colorized = bgr[line_map]
        return colorized

    def accumulate_preds(self, line_strings: List[LineString], image_id: str, pred_json: List[dict]):
        for line in line_strings:
            mask = self._draw_single_line(line, self.thickness)
            mask = (np.all(mask > 0, axis=-1)).astype(np.uint8)
            mask = np.asfortranarray(mask)
            rle = maskUtils.encode(mask)
            if isinstance(rle['counts'], bytes):
                rle['counts'] = rle['counts'].decode('utf-8')
            prediction = {
                "image_id": image_id,
                "category_id": line.class_id,
                "segmentation": rle,
                "score": 1.,
            }
            pred_json.append(prediction)
        return pred_json

    def save_images(self, images_to_save, img_file):
        self._imshow_save.show_imgs(images_to_save)
        save_image = self._imshow_save.update_whole_image()
        filename = img_file.replace('/images/validation', '/result/results')
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        print('save filename:', filename)
        cv2.imwrite(filename, save_image)


def main():
    line_detector = LineStringDetector(cfg.DATA_PATH)
    line_detector.detect_line_strings()

if __name__ == '__main__':
    main()