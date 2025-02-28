import os
import glob
import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from skimage.feature import peak_local_max
from scipy.interpolate import interp1d
from show_imgs import ImageShow


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
    points: np.ndarray = None        # 원본 선상의 샘플링된 점들 (N,2)
    ext_points: np.ndarray = None    # 양쪽으로 확장된 선상의 점들 ((N+M),2)
    src_range: Tuple[int, int] = None  # ext_points 내에서 원래 points가 차지하는 인덱스 범위
    length: float = 0             # 선의 길이 (유클리드 누적거리)


class LineStringDetector:
    peak_thresh = 0.07     # peak에서의 최소값
    peak_min_dist = 3     # peak 사이의 최소 거리 (픽셀)
    id_offset = 10        # peak ID의 최소 오프셋
    sample_stride = 10    # 샘플링 간격 (픽셀)
    extend_len = 30       # 선 확장 길이 (픽셀)

    def __init__(self, data_path: str, npy_path: str):
        self._data_path = data_path
        self._npy_path = npy_path
        self._img_shape = (100, 100)
        self._pallette = [METAINFO[i]['color'] for i in range(len(METAINFO))]
        self._id_count = 0
        self._imshow = ImageShow('images', columns=4, scale=0.6)

    def detect_line_strings(self):
        # data_path 내의 모든 png 이미지에 대해 처리
        file_list = glob.glob(os.path.join(self._npy_path, 'segmap', '*.npy'))
        file_list.sort()
        file_list = file_list[2:]
        for i, file_name in enumerate(file_list):
            print(f'========== file_name: {i} / {file_name}')
            image, seg_map, pred_img = self._read_image(file_name)
            self._img_shape = image.shape[:2]
            self._id_count = self.id_offset
            line_string_list = []
            # seg_img의 채널마다 (예: 차선 클래스별) 처리
            for class_id in range(1, seg_map.shape[-1]):
                print(f'---------- class_id: {class_id}, name: {METAINFO[class_id]["name"]}, color: {METAINFO[class_id]["color"]}')
                if METAINFO[class_id]["name"] in ['lane_line']:
                    print(f'{METAINFO[class_id]["name"]} is ignored')
                    continue
                pred_class_map = np.all(pred_img == METAINFO[class_id]['color'], axis=-1).astype(np.uint8)
                line_map, line_strings = self._thin_image(pred_class_map, class_id)
                if len(line_strings) == 0:
                    print(f'{METAINFO[class_id]["name"]} has no line_strings')
                    continue
                ext_lines = self._extend_lines(line_map, line_strings)
                merged_lines = self._merge_lines(ext_lines)
                print(f'[detect_line_strings] new lines: {len(merged_lines)}, IDs: {[ls.id for ls in merged_lines]}')
                line_string_list.extend(merged_lines)
            
            print(f'>>>>>>>>>> line_string_list: {len(line_string_list)}')
            self._show_line_map(self._draw_line_strings(line_string_list), 'merged_map', dilate=True)

    def _read_image(self, file_name: str):
        seg_map = np.load(file_name)
        img_file = file_name.replace(self._npy_path, self._data_path).replace('segmap', 'images').replace('.npy', '.png')
        image = cv2.imread(img_file)
        pred_file = img_file.replace('/images', '/pred_images')
        pred_img = cv2.imread(pred_file)
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
        self._imshow.show(pred_img, 'pred_img')
        return image, seg_map, pred_img
    
    def _thin_image(self, seg_map: np.ndarray, class_id: int) -> np.ndarray:
        '''
        각 peak를 중심으로 7x7 영역의 median 값을 임계값으로 하여
        해당 blob(connected component)를 flood fill한 후 thinning을 적용.
        thin된 선은 해당 peak에 대응하는 label (k+self.id_offset)로 채워진다.
        '''
        print(f'----- [thin_image] -----')
        line_strings = []
        line_map = np.zeros_like(seg_map, dtype=np.int32)
        line_blobs = np.zeros_like(seg_map, dtype=np.int32)

        y, x = np.nonzero(seg_map)
        self._imshow.show(seg_map*255, 'seg_map', 1)
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
            self._imshow.show(blob_mask, 'blob_mask', 1)

            # cv2.ximgproc.thinning 적용 (얇은 선 추출)
            # (cv2.ximgproc.thinning은 입력이 binary 이미지여야 함)
            line_img = cv2.ximgproc.thinning(blob_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            num_pixels = (line_img > 0).sum()
            if num_pixels < 40:
                print(f'[thin_image] skip small blob at ({x}, {y}), value: {fill_value}, blob size: {num_pixels}')
                continue
            print(f'[thin_image] fill from point at ({x}, {y}), value: {fill_value}, blob size: {num_pixels}')
            # 결과를 line_map에 누적 (겹치는 영역은 덮어쓰기)
            line_map[line_img > 0] = fill_value
            line_strings.append(LineString(id=fill_value, class_id=class_id, peak=(x, y)))
            fill_value += 1
            vis_img = (line_map * 10).astype(np.uint8)
            self._imshow.show(vis_img, 'line_map', dilate=True)
        
        self._imshow.remove(['line_map', 'blob_mask'])
        return line_map, line_strings

    def _extend_lines(self, line_map: np.ndarray, line_strings: List[LineString]) -> List[LineString]:
        '''
        각 label별 thin 선에서 일정 간격의 샘플링 후,
        Hermite spline (여기서는 cubic interpolation)으로 양쪽으로 extend_len 만큼 선을 연장.
        선 길이가 10픽셀 미만인 경우 무시.
        '''
        print(f'----- [extend_lines] -----')
        id_list = np.unique(line_map)
        id_list = id_list[id_list >= self.id_offset]
        for line_string in line_strings:
            # 해당 라벨만 추출한 바이너리 이미지
            line_img = (line_map == line_string.id).astype(np.uint8)
            self._imshow.show(line_img*255, 'line_img', dilate=True)
            line_string.points = self._sample_points(line_img, self.sample_stride)
            if line_string.points.shape[0] < 3:
                line_string.id = None
                continue
            line_string.length = np.sum(np.linalg.norm(np.diff(line_string.points, axis=0), axis=1))
            if line_string.length < 10:
                line_string.id = None
                continue
            line_string = self._extrapolate_line(line_string, self.extend_len, self.sample_stride)

        line_strings = [ls for ls in line_strings if ls.id is not None]
        # 선 길이에 따라 내림차순 정렬
        line_strings.sort(key=lambda ls: ls.length, reverse=True)
        self._imshow.remove(['sample_points', 'ext_point_map', 'line_img'])
        return line_strings

    def _sample_points(self, line_img: np.ndarray, stride: int) -> np.ndarray:
        '''
        line_img 내의 흰색(1) 픽셀을 contour로 추출하고, 그 중 가장 긴 contour에서
        stride 간격으로 점들을 샘플링하여 반환한다.
        '''
        rows, cols = np.nonzero(line_img)
        points = np.stack((cols, rows), axis=1)
        sorted_points = [points[0]]
        direction = points[1] - points[0]
        print(f'[sample_points] start point: {points[0]}, direction: {direction}')
        sorted_points = self._sort_to_direction(points, sorted_points, True, direction, stride)
        sorted_points = self._sort_to_direction(points, sorted_points, False, -direction, stride)
        sorted_points = np.array(sorted_points).astype(np.int32)
        print(f'[sample_points] sorted_points: shape: {sorted_points.shape} \n{sorted_points}')

        point_map = np.zeros_like(line_img, dtype=np.uint8)
        point_map[sorted_points[:, 1], sorted_points[:, 0]] = 255
        cv2.circle(point_map, (sorted_points[0, 0], sorted_points[0, 1]), 3, (255, 0, 255), -1)
        cv2.circle(point_map, (sorted_points[-1, 0], sorted_points[-1, 1]), 3, (255, 0, 255), -1)
        cv2.circle(point_map, (points[0, 0], points[0, 1]), 3, (150, 150, 150), 2)
        self._imshow.show(point_map, 'sample_points', 1, dilate=True)
        return sorted_points.astype(np.int32)
    
    def _sort_to_direction(self, src_points: np.ndarray, sorted_points: List[np.ndarray], to_tail: bool, direction: np.ndarray, stride: int) -> List[np.ndarray]:
        points = src_points.copy()
        while len(points) > 0:
            last_point = sorted_points[-1] if to_tail else sorted_points[0]
            distances = np.sqrt(np.sum((points - last_point)**2, axis=1))
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
            distances = np.sqrt(np.sum((points - last_point)**2, axis=1))
            points = points[distances >= stride]
        return sorted_points

    def _extrapolate_line(self, line_string: LineString, extend_len: int, stride: int) -> LineString:
        '''
        주어진 points(순서대로 연결된 좌표들)에 대해 누적 arc length를 계산한 뒤,
        1D cubic interpolation (interp1d, extrapolate 옵션)을 이용하여
        양 끝으로 extend_len 만큼 선을 연장한 좌표들을 생성한다.
        새로 생성된 선의 좌표에서 원래 points가 차지하는 인덱스 범위(src_range)도 함께 반환.
        '''
        # 누적 arc length 계산
        points = line_string.points
        diffs = np.diff(points, axis=0)
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        s = np.concatenate(([0], np.cumsum(dists)))
        total_length = s[-1]

        # x, y 좌표에 대해 cubic interpolation 생성
        x_interp = interp1d(s, points[:, 0], kind='linear', fill_value="extrapolate")
        y_interp = interp1d(s, points[:, 1], kind='linear', fill_value="extrapolate")

        # stride 픽셀 간격으로 새로운 s 값을 생성 (양쪽 확장)
        new_s = np.arange(-extend_len, total_length + extend_len, stride)
        ext_x = x_interp(new_s)
        ext_y = y_interp(new_s)
        ext_points = np.stack((ext_x, ext_y), axis=-1)
        line_string.ext_points = np.rint(ext_points).astype(np.int32)
        # print('[extrapolate_line] points\n', points)
        print('[extrapolate_line] ext_points\n', ext_points)
        # 원래 s=0와 s=total_length에 해당하는 인덱스를 찾음
        src_start = np.argmin(np.abs(new_s - 0))
        src_end = np.argmin(np.abs(new_s - total_length))
        line_string.src_range = (src_start, src_end)

        # self._img_shape 범위 체크
        valid_mask = (line_string.ext_points[:, 0] >= 0) & (line_string.ext_points[:, 0] < self._img_shape[1]) & \
                    (line_string.ext_points[:, 1] >= 0) & (line_string.ext_points[:, 1] < self._img_shape[0])
        indices = np.where(valid_mask)[0]
        line_string.ext_points = line_string.ext_points[indices]
        line_string.src_range = [line_string.src_range[0] - indices[0], min(line_string.src_range[1] - indices[0], len(line_string.ext_points) - 1)] 
        src_start, src_end = line_string.src_range 
        print(f'src_start: {src_start}, src_end: {src_end}, ext_points shape: {len(line_string.ext_points)}, img_shape: {self._img_shape}')
        point_map = np.zeros(self._img_shape, dtype=np.uint8)
        point_map[line_string.ext_points[:, 1], line_string.ext_points[:, 0]] = 255
        cv2.circle(point_map, (line_string.ext_points[src_start, 0], line_string.ext_points[src_start, 1]), 3, (255, 0, 255), -1)
        cv2.circle(point_map, (line_string.ext_points[src_end, 0], line_string.ext_points[src_end, 1]), 3, (255, 0, 255), -1)
        cv2.circle(point_map, (line_string.ext_points[0, 0], line_string.ext_points[0, 1]), 5, (255, 0, 255), -1)
        cv2.circle(point_map, (line_string.ext_points[-1, 0], line_string.ext_points[-1, 1]), 5, (255, 0, 255), -1)
        self._imshow.show(point_map, 'ext_point_map', 0, dilate=True)
        return line_string

    def _merge_lines(self, line_strings: List[LineString]) -> List[LineString]:
        '''
        dilate된 line_map을 이용하여 각 선의 양쪽 (head, tail)에서 다른 선과
        겹치는 부분이 있는지 확인하고, 겹치면 두 선을 병합한다.
        (병합 시 길이가 더 긴 선의 points를 유지하고, 짧은 선은 제거)
        '''
        print(f'----- [merge_lines] -----')
        # dilate (커널 크기 3x3, 1회 dilation)
        kernel = np.ones((3, 3), np.uint8)

        for line in line_strings:
            if line.id is None:
                continue
            line_map = self._draw_line_strings(line_strings)
            dilated_map = cv2.dilate(line_map.astype(np.uint8), kernel, iterations=2)
            self._show_line_map(dilated_map, 'dilated_map')
            if line.ext_points is None:
                print(f'line {line.id} has no ext_points', line)
                continue
            overlap_ids = self._find_overlap(dilated_map, line.ext_points, line.id)
            if len(overlap_ids) == 0:
                print(f'line {line.id} has no overlap')
                continue
            for overlap_id in overlap_ids:
                print(f'line {overlap_id} is merged to line {line.id}')
                oppo_line = next((l for l in line_strings if l.id == overlap_id), None)
                two_line_map = ((line_map == line.id) | (line_map == oppo_line.id)).astype(np.uint8)
                two_line_map = cv2.dilate(two_line_map, kernel, iterations=2)
                two_line_map = cv2.erode(two_line_map, kernel, iterations=1)
                self._imshow.show(two_line_map*255, 'two_line_map')
                new_line_map = cv2.ximgproc.thinning(two_line_map*255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
                self._imshow.show(new_line_map, 'two_line_map', dilate=True)
                line.points = self._sample_points(new_line_map, 5)
                line.length = np.sum(np.linalg.norm(np.diff(line.points, axis=0), axis=1))
                line = self._extrapolate_line(line, self.extend_len, self.sample_stride)
                oppo_line.id = None
                oppo_line.ext_points = None

        # id가 None이 아닌 선들만 남김
        line_strings = [line for line in line_strings if line.id is not None]
        self._show_line_map(self._draw_line_strings(line_strings), 'merged_map', dilate=True)
        self._imshow.remove(['sample_points', 'ext_point_map', 'merged_map', 'dilated_map', 'two_line_map'])
        return line_strings

    def _draw_line_strings(self, line_strings: List[LineString]):
        image = np.zeros((self._img_shape[1], self._img_shape[0]), dtype=np.uint8)
        for line in line_strings:
            if line.id is None:
                continue
            pts = line.points.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=False, color=(line.id,line.id,line.id), thickness=1)
        return image
    
    def _show_line_map(self, line_map: np.ndarray, window_name: str, dilate: bool = False):
        vis_img = np.zeros_like(line_map, dtype=np.uint8)
        vis_img[line_map > 0] = (line_map[line_map > 0] - self.id_offset) * 20 + 100
        self._imshow.show(vis_img, window_name, 0, dilate)

    def _find_overlap(self, dilated_map: np.ndarray, line_pts: np.ndarray, src_line_id: int) -> int:
        '''
        line_img (binary)와 dilated_map (각 픽셀에 라벨값이 있음)의 겹치는 영역에서,
        배경(0)을 제외하고 가장 많이 등장하는 라벨값을 반환.
        겹치는 부분이 없으면 None 반환.
        '''
        line_img = np.zeros_like(dilated_map, dtype=np.uint8)
        pts = line_pts.reshape((-1, 1, 2))
        cv2.polylines(line_img, [pts], isClosed=False, color=(255,255,255), thickness=1)
        overlap_labels = dilated_map[line_img > 0]
        if overlap_labels.size == 0:
            return None
        unique, counts = np.unique(overlap_labels, return_counts=True)
        print(f'[find_overlap] src_line_id: {src_line_id}, unique: {unique}, counts: {counts}')
        # 배경 0과 자기 자신 제거
        mask = (unique != 0) & (unique != src_line_id) & (counts > 3)
        overlap_ids = unique[mask]
        return overlap_ids



def main():
    npy_path = '/media/dolphin/My Book/mask2former'
    data_path = '/home/dolphin/choi_ws/LaneDetector/dataset/mask2former'
    line_detector = LineStringDetector(data_path, npy_path)
    line_detector.detect_line_strings()


if __name__ == '__main__':
    main()
