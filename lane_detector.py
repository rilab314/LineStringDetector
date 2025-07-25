import os
import glob

import cv2
import numpy as np
from typing import List, Tuple, Set
from dataclasses import dataclass
from skimage.feature import peak_local_max
from scipy.interpolate import interp1d
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
    origin_points: np.ndarray = None  # 기본 segmentation 이미지의 샘플링 포인트 (N,2)
    src_range: Tuple[int, int] = None  # ext_points 내에서 원래 points가 차지하는 인덱스 범위
    length: float = 0  # 선의 길이 (유클리드 누적거리)


class LineStringDetector:
    id_offset = 10  # peak ID의 최소 오프셋
    sample_stride = 10  # 샘플링 간격 (픽셀)
    extend_len = 20  # 선 확장 길이 (픽셀)

    def __init__(self, data_path: str):
        self._data_path = data_path
        self._img_shape = (100, 100)
        self._palette = [info['color'][::-1] for info in METAINFO]
        print('palette:', self._palette)
        self._id_count = 0
        self._imshow_base = ImageShow('base images', columns=3, scale=0.8, enabled=True)
        self._imshow_proc = ImageShow('processing images', columns=3, scale=0.8, enabled=True)
        self._debug_imgs = {}

    def detect_line_strings(self):
        post_processing_path = os.path.join(self._data_path, 'post_processing')
        os.makedirs(post_processing_path, exist_ok=True)
        # data_path 내의 모든 png 이미지에 대해 처리
        file_list = glob.glob(os.path.join(self._data_path, 'images', 'validation', '*.png'))
        file_list.sort()
        file_list = file_list[2:]
        for i, file_name in enumerate(file_list[4:]):
            print(f'===== [file_name] ===== {i} / {len(file_list)} {file_name}')
            image, pred_img = self._read_image(file_name)
            self._img_shape = image.shape[:2]
            self._id_count = self.id_offset

            line_strings = self.extract_lines(pred_img)
            line_strings = self.merge_lines(line_strings, 0)
            line_strings = self.merge_lines(line_strings, 1)
            self._imshow_proc.display(0)

    def _read_image(self, img_file: str):
        image = cv2.imread(img_file)
        print('image file', img_file)
        pred_file = img_file.replace('/images/', '/prediction/')
        pred_img = cv2.imread(pred_file)
        anno_file = img_file.replace('/images/', '/color_annotations/')
        anno_img = cv2.imread(anno_file)
        images = {'image': image, 'GT_img': anno_img, 'pred_img': pred_img}
        self._imshow_base.show_imgs(images)
        return image, pred_img
    
    def extract_lines(self, pred_img: np.ndarray) -> List[LineString]:
        line_string_list = []
        for class_id, color in enumerate(self._palette):
            if class_id == 0:
                continue
            # for class_id in [1, 2, 4, 5, 7, 8, 9]:
            pred_class_map = np.all(pred_img == color, axis=-1).astype(np.uint8)
            line_map, line_strings = self._thin_image(pred_class_map, class_id)
            ext_lines = self._extend_lines(line_map, line_strings)
            line_string_list.extend(ext_lines)
        
        line_img = np.zeros_like(pred_img)
        line_img = self._draw_colored_lines(line_img, line_string_list)
        self._imshow_proc.show(line_img, 'extracted lines')
        return line_string_list
    
    def merge_lines(self, src_line_strings: List[LineString], iter: int) -> List[LineString]:
        dst_line_strings = []
        print(f'=========== [merge_lines] iter={iter}, src_line_strings: {len(src_line_strings)}')
        for class_id, color in enumerate(self._palette):
            if class_id in [0, 3]:
                continue
            class_line_strings = [line for line in src_line_strings if line.class_id == class_id]
            merged_lines = self._merge_lines_by_class(class_line_strings, iter_count=1)
            print(f'[merge_lines] class_id={class_id}, src lines={len(class_line_strings)}, merged={len(merged_lines)}')
            dst_line_strings.extend(merged_lines)

        line_img = np.zeros([self._img_shape[0], self._img_shape[1], 3], dtype=np.uint8)
        line_img = self._draw_colored_lines(line_img, dst_line_strings)
        self._imshow_proc.show(line_img, f'merged_lines_{iter}')
        return dst_line_strings

    def _thin_image(self, seg_map: np.ndarray, class_id: int):
        print(f'----- [thin_image] -----')
        line_strings = []
        line_map = np.zeros_like(seg_map, dtype=np.int32)
        line_blobs = np.zeros_like(seg_map, dtype=np.int32)
        y, x = np.nonzero(seg_map)
        fill_value = self.id_offset
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
            if line_string.points.shape[0] < 3:
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
        head_dir = points[1] - points[0]
        head_dir = head_dir / np.linalg.norm(head_dir)
        tail_dir = points[-1] - points[-2]
        tail_dir = tail_dir / np.linalg.norm(tail_dir)
        n_head = extend_len // stride
        head_ext = np.array([
            points[0] - head_dir * stride * i
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

    def _detect_lines(self, line_strings: List[LineString]) -> List[LineString]:
        origin_center_lines = []
        origin_lane_lines = []
        probe_max_len = 30
        start_len = 5
        tolerance = 1

        for line in line_strings:
            if line.class_id == 1:
                origin_center_lines.append(line)
            if line.class_id == 3:
                origin_lane_lines.append(line)

        center_line_img = self._draw_line_strings(origin_center_lines, origin=True)
        lane_line_img = self._draw_line_strings(origin_lane_lines, origin=True)

        mask = np.any(center_line_img != 0, axis=-1)
        center_line_img[mask] = np.clip(center_line_img[mask].astype(int) + 100, 0, 255).astype(np.uint8)
        mask = np.any(lane_line_img != 0, axis=-1)
        lane_line_img[mask] = np.clip(lane_line_img[mask].astype(int) + 100, 0, 255).astype(np.uint8)

        normal_vector_img = np.zeros_like(center_line_img, dtype=np.uint8)

        for line in origin_center_lines:
            points = line.origin_points
            if len(points) < 2:
                continue
            N = len(points)
            normals = np.zeros_like(points, dtype=np.float32)
            for i in range(N):
                if i == 0:
                    tangent = points[i + 1] - points[i]
                elif i == N - 1:
                    tangent = points[i] - points[i - 1]
                else:
                    tangent = points[i + 1] - points[i - 1]
                norm = np.linalg.norm(tangent)
                if norm < 1e-6:
                    continue
                unit_tangent = tangent / norm
                normal_vec = np.array([-unit_tangent[1], unit_tangent[0]])
                normals[i] = normal_vec

            for i in range(N):
                p = points[i].astype(int)
                n = normals[i]
                if np.linalg.norm(n) < 1e-6:
                    continue
                for side, color in zip([-1, 1], [(255, 200, 0), (0, 255, 255)]):
                    for t in range(start_len, probe_max_len + 1):
                        probe_pt = (p + side * n * t).astype(int)
                        x, y = probe_pt
                        if not (0 <= x < lane_line_img.shape[1] and 0 <= y < lane_line_img.shape[0]):
                            continue
                        roi = lane_line_img[max(0, y - tolerance):y + tolerance + 1,
                              max(0, x - tolerance):x + tolerance + 1]
                        if np.any(roi != 0):
                            cv2.circle(normal_vector_img, (x, y), 2, color, -1)

        origin_img = cv2.addWeighted(center_line_img, 1.0, lane_line_img, 1.0, 0)
        origin_img = cv2.addWeighted(origin_img, 1.0, normal_vector_img, 1.0, 0)
        cv2.imshow('detected_lane_points_by_probe', origin_img)
        cv2.waitKey(0)
        print('======== next line ========')
        return line_strings

    def _merge_lines_by_class(self, line_strings: List[LineString], iter_count: int = 0) -> List[LineString]:
        if len(line_strings) == 0:
            print('[merge_lines_class] line_strings is empty')
            return []
        print(f'----- [merge_lines_class] -----')
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
        label_map = dilated_map if dilated_map.ndim == 2 else dilated_map[:, :, 0]

        orig_lines = self._draw_line_strings(line_strings)
        # self._imshow_proc.show(orig_lines, 'orig_lines', offset=100)
        # self._imshow_proc.show(dilated_map, 'dilated_map', offset=100)

        line_img = np.zeros_like(dilated_map, dtype=np.uint8)
        pts = this_line.ext_points.reshape((-1, 1, 2))
        cv2.polylines(line_img, [pts], isClosed=False, color=(255, 255, 255), thickness=3)
        line_map = line_img[:, :, 0]

        label_map[line_map == 0] = 0
        overlap_labels = label_map[label_map > 0]
        if overlap_labels.size == 0:
            return set()
        unique, counts = np.unique(overlap_labels, return_counts=True)

        mask_cond = (unique != 0) & (unique != src_line_id) & (counts > 3)
        label_ids = set(unique[mask_cond].tolist())
        print(f'unique: {unique}, counts: {counts}, label_ids: {label_ids}')

        line_string_map = dilated_map.copy()
        line_string_map[line_string_map > 0] = 255
        cv2.polylines(line_string_map, [pts], isClosed=False, color=(0, 0, 255), thickness=3)
        # self._imshow_proc.show(line_string_map, 'line_string_map', wait_ms=0)
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

    def _draw_line_strings(self, line_strings: List[LineString], extend=False, origin=False, color=None):
        image = np.zeros((self._img_shape[0], self._img_shape[1], 3), dtype=np.uint8)
        for line in line_strings:
            if line.id is None or line.points is None:
                continue
            if extend:
                pts = line.ext_points.reshape((-1, 1, 2))
            if origin:
                pts = line.origin_points.reshape((-1, 1, 2))
            else:
                pts = line.points.reshape((-1, 1, 2))
            line_color = (line.id, line.id, line.id) if color is None else color
            cv2.polylines(image, [pts], isClosed=False, color=line_color, thickness=2)
        return image

    def _draw_single_line(self, line_string : LineString, extend=False):
        image = np.zeros((self._img_shape[0], self._img_shape[1], 3), dtype=np.uint8)
        if line_string.id is None or line_string.points is None:
            return image
        pts = line_string.points.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=False, color=(line_string.id, line_string.id, line_string.id), thickness=2)
        return image

    def _draw_colored_lines(self, pred_img, line_strings : List[LineString]):
        image = pred_img.copy()
        for line in line_strings:
            if line.id is None:
                continue
            pts = line.points.reshape((-1, 1, 2))
            color = self._palette[line.class_id]
            cv2.polylines(image, [pts], isClosed=False, color=color, thickness=3)
        return image


def main():
    line_detector = LineStringDetector(cfg.DATA_PATH)
    line_detector.detect_line_strings()

if __name__ == '__main__':
    main()