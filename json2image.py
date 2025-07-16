#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from glob import glob

import cv2
import numpy as np

# -------------------------------------------------------------------
# 1) METAINFO 정의
# -------------------------------------------------------------------
METAINFO = [
    {'id': 0,  'name': 'ignore',                        'color': (0,   0,   0  )},
    {'id': 1,  'name': 'center_line',                   'color': (77,  77,  255)},
    {'id': 2,  'name': 'u_turn_zone_line',              'color': (77,  178, 255)},
    {'id': 3,  'name': 'lane_line',                     'color': (77,  255, 77 )},
    {'id': 4,  'name': 'bus_only_lane',                 'color': (255, 153, 77 )},
    {'id': 5,  'name': 'edge_line',                     'color': (255, 77,  77 )},
    {'id': 6,  'name': 'path_change_restriction_line',  'color': (178, 77,  255)},
    {'id': 7,  'name': 'no_parking_stopping_line',      'color': (77,  255, 178)},
    {'id': 8,  'name': 'guiding_line',                  'color': (255, 178, 77 )},
    {'id': 9,  'name': 'stop_line',                     'color': (77,  102, 255)},
    {'id': 10, 'name': 'safety_zone',                   'color': (255, 77,  128)},
    {'id': 11, 'name': 'bicycle_lane',                  'color': (128, 255, 77 )},
]
# name 으로 METAINFO 조회할 수 있게 dict 생성
meta_by_name = {m['name']: m for m in METAINFO}


def json_to_image(json_path, output_dir, canvas_size=(768, 768), thickness=2):
    """
    한 개의 JSON 파일을 읽어 768x768 크기의 이미지로 변환하여 output_dir에 저장.
    """
    # JSON 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # JSON 최상위가 dict인지 list인지 판별하여 RoadObject 목록 추출
    if isinstance(data, dict):
        objs = data.get('RoadObject', [])
    elif isinstance(data, list):
        objs = data
    else:
        print(f"[Warning] unexpected JSON structure: {json_path}, skipping.")
        return

    # 모든 image_points 좌표 모아서 원본 이미지 크기(가정) 계산
    all_pts = [pt for obj in objs for pt in obj.get('image_points', [])]
    if all_pts:
        xs, ys = zip(*all_pts)
        orig_w, orig_h = max(xs), max(ys)
        # 스케일 비율 (가로, 세로)
        sx = canvas_size[0] / orig_w
        sy = canvas_size[1] / orig_h
    else:
        sx = sy = 1.0

    # 빈 캔버스 생성 (BGR, uint8)
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)

    # 각 객체 그리기
    for obj in objs:
        cat = obj.get('category')
        meta = meta_by_name.get(cat)
        if not meta:
            continue  # METAINFO에 정의되지 않은 카테고리 스킵

        pts = obj.get('image_points', [])
        if len(pts) < 2:
            continue

        # 스케일링 후 정수 좌표 배열 생성
        pts_scaled = np.array(
            [[int(x * sx), int(y * sy)] for x, y in pts],
            dtype=np.int32
        ).reshape((-1, 1, 2))

        # 폴리라인 그리기
        cv2.polylines(
            img=canvas,
            pts=[pts_scaled],
            isClosed=False,
            color=meta['color'],
            thickness=thickness
        )
    cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR, canvas)
    # 파일명 기반으로 저장
    base = os.path.splitext(os.path.basename(json_path))[0]
    out_path = os.path.join(output_dir, base + '.png')
    cv2.imwrite(out_path, canvas)
    print(f"Saved: {out_path}")


def main():
    # -------------------------------------------------------------------
    # 2) 입력/출력 폴더 설정 (환경에 맞게 수정!)
    # -------------------------------------------------------------------
    input_dir  = '/home/humpback/youn_ws/Internimage_rilab/data/satellite_good_matching_241125/label'
    output_dir = '/home/humpback/youn_ws/Internimage_rilab/data/satellite_good_matching_241125/vis768'
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------
    # 3) 모든 JSON 파일 순회하여 변환
    # -------------------------------------------------------------------
    pattern = os.path.join(input_dir, '*.json')
    for json_path in glob(pattern):
        json_to_image(json_path, output_dir)


if __name__ == '__main__':
    main()
