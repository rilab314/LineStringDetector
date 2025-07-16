import os
import glob

import cv2
import numpy as np
from typing import List, Tuple, Set
from dataclasses import dataclass
from skimage.feature import peak_local_max
from scipy.interpolate import interp1d
from show_imgs import ImageShow

METAINFO = [
    {'id': 0, 'name': 'ignore', 'color': (0, 0, 0)},
    {'id': 1, 'name': 'center_line', 'color': (77, 77, 255)},
    {'id': 2, 'name': 'u_turn_zone_line', 'color': (77, 178, 255)},
    {'id': 3, 'name': 'lane_line', 'color': (77, 255, 77)}, # 제외
    {'id': 4, 'name': 'bus_only_lane', 'color': (255, 153, 77)}, # 제외
    {'id': 5, 'name': 'edge_line', 'color': (255, 77, 77)}, # 제외
    {'id': 6, 'name': 'path_change_restriction_line', 'color': (178, 77, 255)},
    {'id': 7, 'name': 'no_parking_stopping_line', 'color': (77, 255, 178)},
    {'id': 8, 'name': 'guiding_line', 'color': (255, 178, 77)}, # 제외
    {'id': 9, 'name': 'stop_line', 'color': (77, 102, 255)},
    {'id': 10, 'name': 'safety_zone', 'color': (255, 77, 128)}, # 제외
    {'id': 11, 'name': 'bicycle_lane', 'color': (128, 255, 77)}, # 제외
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
    peak_thresh = 0.07  # peak에서의 최소값
    peak_min_dist = 3  # peak 사이의 최소 거리 (픽셀)
    id_offset = 10  # peak ID의 최소 오프셋
    sample_stride = 10  # 샘플링 간격 (픽셀)
    extend_len = 20  # 선 확장 길이 (픽셀)

    def __init__(self, data_path: str, npy_path: str):
        self._data_path = data_path
        self._npy_path = npy_path
        self._img_shape = (100, 100)
        self._pallette = [METAINFO[i]['color'] for i in range(len(METAINFO))]
        self._id_count = 0
        self._imshow = ImageShow('images', columns=4, scale=0.6)
        self._post_procssing_path = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2025_LaneDetector/post_processing_inference'

    def detect_line_strings(self):
        # data_path 내의 모든 png 이미지에 대해 처리
        file_list = glob.glob(os.path.join(self._npy_path, 'segmap', '*.npy'))
        file_list.sort()
        file_list = file_list[2:]
        for i, file_name in enumerate(file_list):
            base = os.path.splitext(os.path.basename(file_name))[0]
            png_name = base + '.png'
            image, seg_map, pred_img = self._read_image(file_name)
            self._img_shape = image.shape[:2]
            self._id_count = self.id_offset
            save_img = np.zeros((self._img_shape[0], self._img_shape[1], 3), dtype=np.uint8)
            line_string_list = []
            # seg_img의 채널마다 (예: 차선 클래스별) 처리
            for class_id in range(1, seg_map.shape[-1]):
            # for class_id in [1, 2, 4, 5, 7, 8, 9]:
                pred_class_map = np.all(pred_img == METAINFO[class_id]['color'], axis=-1).astype(np.uint8)
                line_map, line_strings = self._thin_image(pred_class_map, class_id)
                ext_lines = self._extend_lines(line_map, line_strings)
                merged_lines = self._merge_lines(ext_lines)
                line_string_list.extend(merged_lines)
            save_img = self._draw_class_id(save_img, line_string_list)
            os.makedirs(self._post_procssing_path, exist_ok=True)
            cv2.imwrite(os.path.join(self._post_procssing_path, png_name), save_img)

    def _read_image(self, file_name: str):
        seg_map = np.load(file_name)
        img_file = file_name.replace(self._npy_path, self._data_path).replace('segmap', 'images').replace('.npy',
                                                                                                          '.png')
        image = cv2.imread(img_file)
        pred_file = img_file.replace('/images', '/pred_images')
        pred_img = cv2.imread(pred_file)
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
        return image, seg_map, pred_img

    def _thin_image(self, seg_map: np.ndarray, class_id: int) -> np.ndarray:
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
            num_pixels = (line_img > 0).sum()

            # 결과를 line_map에 누적 (겹치는 영역은 덮어쓰기)
            line_map[line_img > 0] = fill_value
            line_strings.append(LineString(id=fill_value, class_id=class_id, peak=(x, y)))
            fill_value += 1
            vis_img = (line_map * 10).astype(np.uint8)
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
        # 1) 머리 방향 벡터(head_dir)와 꼬리 방향 벡터(tail_dir) 계산
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

    def _merge_lines(self, line_strings: List[LineString]) -> List[LineString]:
        print(f'----- [merge_lines] -----')
        # dilate (커널 크기 3x3, 1회 dilation)
        kernel = np.ones((3, 3), np.uint8)
        line_map = self._draw_line_strings_extend(line_strings)
        dilated_map = cv2.dilate(line_map.astype(np.uint8), kernel, iterations=2)

        debug_img = dilated_map.copy()
        for line in line_strings:
            # 6) 선 중앙에 ID 표시
            #    boundingRect 로 바운딩 박스 구하고, 그 중간 좌표에 텍스트
            bx, by, bw, bh = cv2.boundingRect(line.points)
            cx, cy = bx + bw // 2, by + bh // 2
            cv2.putText(
                debug_img,
                f"{line.id}",
                org=(cx, cy),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 255, 255),
                thickness=2
            )

        overlap_debug = []
        neighbor_debug = []
        for line in line_strings:
            overlap_ids = self._find_overlap(dilated_map, line, line.id, debug_img)
            overlap_debug.extend(overlap_ids)
            if len(overlap_ids) == 0:
                continue
            for overlap_id in overlap_ids:
                oppo_line = next((l for l in line_strings if l.id == overlap_id), None)
                if oppo_line is None:
                    continue
                if oppo_line.points is None:
                    continue
                combined_img = self._connect_tail2head(line, oppo_line)

                oppo_img = self._draw_line_string(oppo_line)
                oppo_img[oppo_img > 0] = np.clip(oppo_img[oppo_img > 0].astype(int) + 100, 0,
                                                         255).astype(oppo_img.dtype)
                # cv2.imshow('oppo_line', oppo_img)
                line_img = self._draw_line_string(line)
                line_img[line_img > 0] = np.clip(line_img[line_img > 0].astype(int) + 100, 0,
                                                         255).astype(line_img.dtype)
                # cv2.imshow('line_line', line_img)

                combined_img[combined_img > 0] = np.clip(combined_img[combined_img > 0].astype(int) + 100, 0, 255).astype(combined_img.dtype)
                # cv2.imshow('combined_img', combined_img)
                # cv2.waitKey(0)
                # cv2.destroyWindow('oppo_line')
                # cv2.destroyWindow('line_line')
                # cv2.destroyWindow('combined_img')

                combined_img = cv2.cvtColor(combined_img, cv2.COLOR_BGR2GRAY)
                line.points = self._sample_points(combined_img, self.sample_stride)
                line.length = np.sum(np.linalg.norm(np.diff(line.points, axis=0), axis=1))
                line = self._extrapolate_line(line, self.extend_len, self.sample_stride)
                oppo_line.id = None
                oppo_line.ext_points = None
                print(f'=======overlap_debug=======\n{overlap_debug}')

        merged_img = self._draw_line_strings(line_strings)
        merged_img[merged_img > 0] = np.clip(merged_img[merged_img > 0].astype(int) + 100, 0, 255).astype(
            merged_img.dtype)
        # cv2.imshow('merged_img', merged_img)
        # cv2.waitKey(0)
        # cv2.destroyWindow('neighbor_debug')
        # cv2.destroyWindow('merged_img')

        # id가 None이 아닌 선들만 남김
        line_strings = [line for line in line_strings if line.id is not None]
        return line_strings

    def _find_overlap(self, dilated_map: np.ndarray, line: LineString, src_line_id: int, debug_img: np.ndarray) -> Set[int]:
        # 0) ext_points 유무 체크
        if line.ext_points is None:
            return set()
        # 1) 원본 선분 바이너리 마스크 생성
        line_pts = line.points
        line_ext_pts = line.ext_points
        line_img = np.zeros_like(dilated_map, dtype=np.uint8)
        pts = line_pts.reshape((-1, 1, 2))
        cv2.polylines(line_img, [pts], isClosed=False, color=(255, 255, 255), thickness=3)
        debug_vis = debug_img.copy()
        mask = debug_vis > 0
        debug_vis[mask] = np.clip(debug_vis[mask].astype(int) + 100, 0, 255).astype(debug_vis.dtype)
        label_map = dilated_map if dilated_map.ndim == 2 else dilated_map[:, :, 0]

        blue_mask = (label_map == src_line_id)
        red_mask = (line_img[:, :, 0] > 0)
        debug_vis[blue_mask] = (255, 0, 0)
        debug_vis[red_mask] = (0, 0, 255)

        overlap_labels_red = label_map[red_mask]
        if overlap_labels_red.size > 0:
            unique, counts = np.unique(overlap_labels_red, return_counts=True)
            for i, (u, c) in enumerate(zip(unique, counts)):
                text = f"id:{int(u)}/cnt:{int(c)}"
                cv2.putText(
                    debug_vis, text,
                    org=(5, 20 + i * 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(255, 255, 0),
                    thickness=1
                )

        # cv2.imshow('overlap_debug', debug_vis)
        # cv2.waitKey(0)

        if overlap_labels_red.size == 0:
            base_ids = set()
        else:
            unique, counts = np.unique(overlap_labels_red, return_counts=True)
            mask_cond = (unique != 0) & (unique != src_line_id) & (counts > 3)
            base_ids = set(unique[mask_cond].tolist())
        return base_ids

    def _connect_tail2head(self, line, oppo_line):
        image = np.zeros((self._img_shape[1], self._img_shape[0], 3), dtype=np.uint8)
        color = (line.id, line.id, line.id)
        line_pts = line.points.reshape((-1, 1, 2))
        oppo_pts = oppo_line.points.reshape((-1, 1, 2))
        cv2.polylines(image, [line_pts], isClosed=False, color=color, thickness=2)
        cv2.polylines(image, [oppo_pts], isClosed=False, color=color, thickness=2)

        head1 = tuple(line.points[0].astype(int))
        tail1 = tuple(line.points[-1].astype(int))
        head2 = tuple(oppo_line.points[0].astype(int))
        tail2 = tuple(oppo_line.points[-1].astype(int))

        d1 = np.hypot(tail1[0] - head2[0], tail1[1] - head2[1])
        d2 = np.hypot(head1[0] - tail2[0], head1[1] - tail2[1])

        if d1 <= d2:
            start_pt, end_pt = tail1, head2
        else:
            start_pt, end_pt = head1, tail2

        cv2.line(image, start_pt, end_pt, color=color, thickness=2)
        return image

    def _find_neighboring_lanes(self, dilated_map: np.ndarray, line: LineString) -> Set[int]:
        if line.ext_points is None:
            return set()

        label_map = dilated_map if dilated_map.ndim == 2 else dilated_map[:, :, 0]
        h, w = self._img_shape
        neighbors = set()

        debug_img = (
            cv2.cvtColor(dilated_map, cv2.COLOR_GRAY2BGR)
            if dilated_map.ndim == 2
            else dilated_map.copy()
        )
        cv2.putText(
            debug_img, f"Line ID: {line.id}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA
        )
        for pt in line.ext_points:
            cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)

        kernel_size = 60
        half = kernel_size // 2
        endpoints = [line.ext_points[0], line.ext_points[-1]]

        for idx, (xf, yf) in enumerate(endpoints):
            x, y = int(xf), int(yf)
            x1 = max(0, x - half);  x2 = min(w - 1, x + half)
            y1 = max(0, y - half);  y2 = min(h - 1, y + half)

            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

            region = label_map[y1:y2+1, x1:x2+1]
            ids = [int(uid) for uid in np.unique(region) if uid != 0 and uid != line.id]
            neighbors.update(ids)

            text = "IDs:" + (",".join(map(str, ids)) or "None")
            pos = (x2 + 5, y1 + 15 + idx * 20)
            cv2.putText(
                debug_img, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
            )

        debug_img[debug_img > 0] = np.clip(debug_img[debug_img > 0].astype(int) + 100, 0, 255).astype(
            debug_img.dtype)
        # cv2.imshow('neighbor_debug', debug_img)
        # cv2.waitKey(0)

        return neighbors


    def _draw_line_strings(self, line_strings: List[LineString], extend=False):
        image = np.zeros((self._img_shape[0], self._img_shape[1], 3), dtype=np.uint8)
        for line in line_strings:
            if line.id is None or line.points is None:
                continue
            pts = line.points.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=False, color=(line.id, line.id, line.id), thickness=2)
        return image

    def _draw_line_strings_extend(self, line_strings: List[LineString], extend=False):
        image = np.zeros((self._img_shape[0], self._img_shape[1], 3), dtype=np.uint8)
        for line in line_strings:
            if line.id is None or line.points is None:
                continue
            pts = line.ext_points.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=False, color=(line.id, line.id, line.id), thickness=2)
        return image

    def _draw_line_string(self, line_string : LineString, extend=False):
        image = np.zeros((self._img_shape[0], self._img_shape[1], 3), dtype=np.uint8)
        if line_string.id is None or line_string.points is None:
            return image
        pts = line_string.points.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=False, color=(line_string.id, line_string.id, line_string.id), thickness=2)
        return image

    def _draw_class_id(self, pred_img, line_strings : List[LineString]):
        image = pred_img.copy()
        for line in line_strings:
            if line.id is None:
                continue
            pts = line.points.reshape((-1, 1, 2))
            color = METAINFO[line.class_id]['color']
            cv2.polylines(image, [pts], isClosed=False, color=color, thickness=3)
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR, image)
        return image


def main():
    npy_path = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2025_LaneDetector/mask2former'
    data_path = '/home/humpback/youn_ws/LaneDetector_rilab/dataset/mask2former'
    line_detector = LineStringDetector(data_path, npy_path)
    line_detector.detect_line_strings()


if __name__ == '__main__':
    main()