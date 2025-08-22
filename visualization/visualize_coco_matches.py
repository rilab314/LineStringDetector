#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COCO GT vs Pred 시각화 (TP/FP/FN)
- iouType='segm' 또는 'bbox' 지원
- IoU≥thr로 그리디 매칭 → TP/FP/FN 색 분리 그리기
- segm: RLE/poly 모두 지원 (RLE 우선 디코딩)
- bbox: 사각형 그리기
- 저장: 원본/겹침/마스크 오버레이 3종(선택) 저장

사용 전 상단의 PATH/옵션을 채우세요.
"""

import os
import json
import cv2
import numpy as np
from typing import List, Dict, Tuple, Union
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

# ------------- 설정 (수정하세요) -------------
GT_JSON   = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2025_LaneDetector/new_coco_dataset/annotations/instances_validation2017.json'
PRED_JSON = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2025_LaneDetector/ade20k/satellite_ade20k_250820/process/coco_pred_instances_merged.json'
IMG_ROOT  = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2025_LaneDetector/ade20k/satellite_ade20k_250820/images/validation'  # GT images[].file_name 기준으로 찾음
OUT_DIR   = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2025_LaneDetector/ade20k/satellite_ade20k_250820/coco_visualization'
os.makedirs(OUT_DIR, exist_ok=True)

IOU_TYPE    = "segm"     # "segm" or "bbox"
IOU_THR     = 0.50
SCORE_THR   = 0.00       # 낮게 두고 많이 보거나, 0.2/0.5 등으로 필터
MAX_DETS    = 100        # 한 이미지당 그릴 pred 최대 개수
DRAW_FILLED = True       # True면 반투명 채움, False면 외곽선만
ALPHA       = 0.4        # 채우기일 때 투명도
LINE_THICK  = 2
RANDOM_SUBSAMPLE = 0     # 0이면 전체, >0이면 매 N장마다 1장만 저장
# -------------------------------------------


def _rle_from_any(segm, h, w):
    """segmentation -> RLE(dict). list(poly)면 merge, dict면 보정."""
    if isinstance(segm, dict) and 'counts' in segm and 'size' in segm:
        rle = segm.copy()
        if isinstance(rle['counts'], bytes):
            rle['counts'] = rle['counts'].decode('utf-8')
        return rle
    # polygon(list or list-of-lists)
    if isinstance(segm, list):
        polys = segm
        if len(polys) > 0 and all(isinstance(v, (int, float)) for v in polys):
            polys = [polys]
        valid = []
        for p in polys:
            if isinstance(p, list) and len(p) >= 6 and len(p) % 2 == 0:
                valid.append(p)
        if not valid:
            return None
        rles = maskUtils.frPyObjects(valid, h, w)
        rle = maskUtils.merge(rles) if isinstance(rles, list) else rles
        if isinstance(rle['counts'], bytes):
            rle['counts'] = rle['counts'].decode('utf-8')
        return rle
    return None


def _iou_matrix_segm(gt_anns, dt_anns, h, w):
    """세그멘테이션 IoU 행렬 (len(dt), len(gt))"""
    if len(gt_anns) == 0 or len(dt_anns) == 0:
        return np.zeros((len(dt_anns), len(gt_anns)), dtype=np.float32)
    gt_rles = []
    for g in gt_anns:
        r = _rle_from_any(g['segmentation'], h, w)
        if r is None:
            # 폴백: bbox로 마스크
            x, y, bw, bh = g['bbox']
            mask = np.zeros((h, w), dtype=np.uint8, order='F')
            x2 = min(int(round(x + bw)), w)
            y2 = min(int(round(y + bh)), h)
            mask[int(round(y)):y2, int(round(x)):x2] = 1
            r = maskUtils.encode(mask)
            r['counts'] = r['counts'].decode('utf-8')
        gt_rles.append(r)
    dt_rles = []
    for d in dt_anns:
        r = _rle_from_any(d['segmentation'], h, w)
        if r is None:
            x, y, bw, bh = d['bbox']
            mask = np.zeros((h, w), dtype=np.uint8, order='F')
            x2 = min(int(round(x + bw)), w)
            y2 = min(int(round(y + bh)), h)
            mask[int(round(y)):y2, int(round(x)):x2] = 1
            r = maskUtils.encode(mask)
            r['counts'] = r['counts'].decode('utf-8')
        dt_rles.append(r)
    ious = maskUtils.iou(dt_rles, gt_rles, [0]*len(gt_rles))  # iscrowd=0
    return np.array(ious, dtype=np.float32)


def _iou_matrix_bbox(gt_anns, dt_anns):
    """바운딩박스 IoU 행렬 (len(dt), len(gt))"""
    def _iou(b1, b2):
        x1, y1, w1, h1 = b1; x2, y2, w2, h2 = b2
        xa1, ya1, xa2, ya2 = x1, y1, x1+w1, y1+h1
        xb1, yb1, xb2, yb2 = x2, y2, x2+w2, y2+h2
        ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
        ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
        iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
        inter = iw*ih
        area1 = w1*h1; area2 = w2*h2
        union = area1 + area2 - inter + 1e-6
        return inter/union
    if len(gt_anns) == 0 or len(dt_anns) == 0:
        return np.zeros((len(dt_anns), len(gt_anns)), dtype=np.float32)
    ious = np.zeros((len(dt_anns), len(gt_anns)), dtype=np.float32)
    for i, d in enumerate(dt_anns):
        for j, g in enumerate(gt_anns):
            ious[i, j] = _iou(d['bbox'], g['bbox'])
    return ious


def _greedy_match(ious: np.ndarray, thr: float):
    """단순 그리디 매칭: dt 우선, 각 dt는 최고 IoU gt와 매칭(중복 매칭 금지)"""
    D, G = ious.shape
    gt_used = np.zeros(G, dtype=bool)
    matches = []
    for di in range(D):
        gj = int(np.argmax(ious[di])) if G > 0 else -1
        iou = ious[di, gj] if G > 0 else 0.0
        if G > 0 and iou >= thr and not gt_used[gj]:
            matches.append((di, gj, float(iou)))
            gt_used[gj] = True
        else:
            matches.append((di, -1, float(iou)))  # FP
    fns = [gj for gj in range(G) if not gt_used[gj]]  # 남은 GT = FN
    return matches, fns


def _colorize(image, mask, color_bgr, alpha=0.4, fill=True, thickness=2):
    if mask.ndim == 2:
        m = (mask > 0).astype(np.uint8)
    else:
        m = mask
    if fill:
        overlay = image.copy()
        overlay[m > 0] = color_bgr
        cv2.addWeighted(overlay, alpha, image, 1-alpha, 0, dst=image)
    else:
        cnts, _ = cv2.findContours((m>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, cnts, -1, color_bgr, thickness)


def _ann_to_mask(ann, h, w, iouType):
    if iouType == 'segm':
        r = _rle_from_any(ann['segmentation'], h, w)
        if r is None:
            # 폴백: bbox 사각형
            x, y, bw, bh = ann['bbox']
            mask = np.zeros((h, w), dtype=np.uint8)
            x2 = min(int(round(x + bw)), w); y2 = min(int(round(y + bh)), h)
            mask[int(round(y)):y2, int(round(x)):x2] = 1
            return mask
        return maskUtils.decode(r)
    else:
        x, y, bw, bh = ann['bbox']
        mask = np.zeros((h, w), dtype=np.uint8)
        x2 = min(int(round(x + bw)), w); y2 = min(int(round(y + bh)), h)
        mask[int(round(y)):y2, int(round(x)):x2] = 1
        return mask


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    coco_gt = COCO(GT_JSON)                                 # GT 로드
    coco_dt = coco_gt.loadRes(PRED_JSON)                    # Pred 로드(파일 경로 허용)

    img_ids = coco_gt.getImgIds()
    for idx, img_id in enumerate(img_ids):
        if RANDOM_SUBSAMPLE and (idx % RANDOM_SUBSAMPLE != 0):
            continue

        im = coco_gt.loadImgs([img_id])[0]
        file_name = im['file_name']
        h, w = int(im['height']), int(im['width'])
        img_path = os.path.join(IMG_ROOT, file_name)
        src = cv2.imread(img_path)
        if src is None:
            # 이미지가 없으면 빈 배경 생성
            src = np.zeros((h, w, 3), dtype=np.uint8)

        # 이 이미지의 GT/DT 어노테이션 수집
        ann_ids_gt = coco_gt.getAnnIds(imgIds=[img_id])
        ann_ids_dt = coco_dt.getAnnIds(imgIds=[img_id])
        gts = coco_gt.loadAnns(ann_ids_gt)
        dts_all = coco_dt.loadAnns(ann_ids_dt)
        # 점수/개수 필터
        dts = sorted([d for d in dts_all if d.get('score', 1.0) >= SCORE_THR],
                     key=lambda d: -d.get('score', 0.0))[:MAX_DETS]

        # 카테고리별로 매칭해도 되지만, 우선 전체 섞어서 단순 IoU 매칭
        if IOU_TYPE == 'segm':
            ious = _iou_matrix_segm(gts, dts, h, w)
        else:
            ious = _iou_matrix_bbox(gts, dts)

        matches, fn_idx = _greedy_match(ious, IOU_THR)  # (dt_i, gt_j|-1, iou)

        # 시각화 캔버스
        vis = src.copy()
        # GT 먼저 그리기 → 얇은 파랑(FN 후보들 포함)
        for j, g in enumerate(gts):
            m = _ann_to_mask(g, h, w, IOU_TYPE)
            _colorize(vis, m, (255, 160, 0) if j in fn_idx else (255, 255, 0),   # 미탐(FN)=파랑, 매칭된 GT는 연노랑로 살짝
                      alpha=ALPHA, fill=DRAW_FILLED, thickness=LINE_THICK)

        # Pred 그리기 (TP=초록, FP=빨강)
        for (di, gj, iou) in matches:
            d = dts[di]
            m = _ann_to_mask(d, h, w, IOU_TYPE)
            if gj >= 0:
                _colorize(vis, m, (0, 220, 0), alpha=ALPHA, fill=DRAW_FILLED, thickness=LINE_THICK)   # TP=green
                # IoU 텍스트
                if d.get('score', None) is not None:
                    cv2.putText(vis, f"{iou:.2f}/{d['score']:.2f}", (int(d['bbox'][0]), int(d['bbox'][1])-2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 0), 1, cv2.LINE_AA)
            else:
                _colorize(vis, m, (0, 0, 255), alpha=ALPHA, fill=DRAW_FILLED, thickness=LINE_THICK)   # FP=red
                if d.get('score', None) is not None:
                    cv2.putText(vis, f"{d['score']:.2f}", (int(d['bbox'][0]), int(d['bbox'][1])-2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 180), 1, cv2.LINE_AA)

        # 저장
        base = os.path.splitext(os.path.basename(file_name))[0]
        out_path = os.path.join(OUT_DIR, f"{base}_vis_{IOU_TYPE}.png")
        cv2.imwrite(out_path, vis)
        print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()
