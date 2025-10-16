# make_table2.py
# 목적: class별 성능 표(table_2)를 CSV로 저장
#   cols: ["class", "AP20 (long)", "pixel accuracy", "# pred instances", "# GT instances"]
#   rows: METAINFO에서 guiding_line(8), safety_zone(10) 제외한 각 클래스
#
# 정의:
#   - AP20 (long): eval_coco_map 방식 그대로 COCO(gt).loadRes(pred) 사용,
#                  IoU=0.20 단일 임계값에서 카테고리별 AP 계산 (MERGED_JSON_PATH vs COCO_ANNO_PATH)
#   - pixel accuracy (per-class): MERGED_JSON_PATH를 per-pixel 라벨맵으로 렌더링 후,
#                  GT 라벨(LABEL_PATH)과 비교하여 각 클래스별(=GT 기준) 픽셀 정확도 TP_c / (GT_c_pixels)
#                  (ignore=0은 제외)
#   - # pred instances: MERGED_JSON_PATH 내 카테고리별 어노테이션 수
#   - # GT instances:   COCO_ANNO_PATH 내 카테고리별 어노테이션 수
#
# 출력:
#   cfg.RESULT_PATH / "Table" / "table_2.csv"

import os
import glob
import json
import csv
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import config as cfg

# ===================== 경로/출력 ===================== #
LABEL_DIR   = cfg.LABEL_PATH
PRED_DIR    = cfg.PRED_PATH  # (table2에서는 직접 사용하지 않지만 유지)
COCO_GT_JSON            = cfg.COCO_ANNO_PATH
PRED_JSON               = cfg.MERGED_JSON_PATH  # merged 예측
OUT_DIR = os.path.join(cfg.RESULT_PATH, "Table")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "table_2.csv")

# ===== 옵션 =====
USE_TQDM       = True
VIS_ENABLE     = False              # 필요 시 True로 바꾸고 시각화 저장 가능하게 확장해도 됨
VIS_N_SAMPLES  = 0
VIS_SAVE_DIR   = os.path.join(OUT_DIR, "table2_visuals")
if VIS_ENABLE:
    os.makedirs(VIS_SAVE_DIR, exist_ok=True)

# ===================== 클래스 팔레트/META ===================== #
METAINFO = [
    {'id': 0,  'name': 'ignore',                        'color': (0, 0, 0)},        # RGB
    {'id': 1,  'name': 'center_line',                   'color': (77, 77, 255)},
    {'id': 2,  'name': 'u_turn_zone_line',              'color': (77, 178, 255)},
    {'id': 3,  'name': 'lane_line',                     'color': (77, 255, 77)},
    {'id': 4,  'name': 'bus_only_lane',                 'color': (255, 153, 77)},
    {'id': 5,  'name': 'edge_line',                     'color': (255, 77, 77)},
    {'id': 6,  'name': 'path_change_restriction_line',  'color': (178, 77, 255)},
    {'id': 7,  'name': 'no_parking_stopping_line',      'color': (77, 255, 178)},
    {'id': 8,  'name': 'guiding_line',                  'color': (255, 178, 77)},
    {'id': 9,  'name': 'stop_line',                     'color': (77, 102, 255)},
    {'id': 10, 'name': 'safety_zone',                   'color': (255, 77, 128)},
    {'id': 11, 'name': 'bicycle_lane',                  'color': (128, 255, 77)},
]
ID2BGR: Dict[int, Tuple[int,int,int]] = {c['id']: (c['color'][2], c['color'][1], c['color'][0]) for c in METAINFO}
BGR2ID: Dict[Tuple[int,int,int], int] = {v: k for k, v in ID2BGR.items()}

# 평가 대상 클래스 (ignore=0, guiding=8, safety=10 제외)
EVAL_CLASS_IDS = [c['id'] for c in METAINFO if c['id'] not in (0, 8, 10)]
ID2NAME = {c['id']: c['name'] for c in METAINFO}

# ===================== 공통 유틸 ===================== #
IGNORE_IDS = {0}

def list_basenames(folder: str, exts=(".png",".PNG",".jpg",".JPG",".jpeg",".JPEG")) -> set:
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, f"*{e}")))
    return set(os.path.splitext(os.path.basename(p))[0] for p in files)

def find_with_any_ext(folder: str, stem: str) -> str:
    for e in (".png",".PNG",".jpg",".JPG",".jpeg",".JPEG"):
        p = os.path.join(folder, stem + e)
        if os.path.exists(p):
            return p
    return ""

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def to_label_index_image(img: np.ndarray, *, is_gt: bool) -> np.ndarray:
    """
    (H,W) 또는 (H,W,1): 인덱스
    (H,W,3): 팔레트(BGR)→인덱스
    GT만 -1 시프트(1-based→0-based)
    """
    if img is None:
        return None
    if img.ndim == 2:
        lab = img.astype(np.int32)
    elif img.ndim == 3 and img.shape[2] == 1:
        lab = img[..., 0].astype(np.int32)
    elif img.ndim == 3 and img.shape[2] == 3:
        h, w = img.shape[:2]
        lab = np.zeros((h, w), dtype=np.int32)
        for cid, bgr in ID2BGR.items():
            mask = np.all(img == np.array(bgr, dtype=np.uint8), axis=-1)
            lab[mask] = int(cid)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    if is_gt:
        lab = lab - 1
        lab[lab < 0] = 0
    return lab

def valid_mask(gt_lab: np.ndarray) -> np.ndarray:
    vm = np.ones_like(gt_lab, dtype=bool)
    for ig in IGNORE_IDS:
        vm &= (gt_lab != ig)
    return vm

# ===================== AP20 (class-wise, eval_coco_map 방식) ===================== #
def ap20_per_class_evalstyle(gt_json: str, pred_json: str) -> Dict[int, float]:
    """
    COCOeval로 IoU=0.20 단일 설정 후, precision에서 카테고리별 AP 계산.
    반환: {cat_id: ap20}
    """
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(pred_json)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
    coco_eval.params.iouThrs = np.array([0.20], dtype=np.float32)
    coco_eval.evaluate()
    coco_eval.accumulate()

    # precision: [T, R, K, A, M] -> T=1에서 [R, K, A, M]
    p = coco_eval.eval['precision']
    if p.size == 0:
        return {}

    p0 = p[0]  # [R, K, A, M]
    # 카테고리 id 순서는 params.catIds 기준
    cat_ids = coco_eval.params.catIds
    ap_by_cat: Dict[int, float] = {}
    for kidx, cid in enumerate(cat_ids):
        # area=all(0), maxDets=100(2)
        pk = p0[:, kidx, 0, 2]
        valid = pk[pk > -1]
        ap_by_cat[int(cid)] = float(np.mean(valid)) if valid.size else float('nan')
    return ap_by_cat

# ===================== 인스턴스 개수 집계 ===================== #
def count_pred_instances_by_cat(pred_json_path: str) -> Dict[int, int]:
    data = load_json(pred_json_path)
    anns = data["annotations"] if isinstance(data, dict) and "annotations" in data else data
    counts: Dict[int, int] = {}
    for a in anns:
        cid = int(a.get("category_id", -1))
        counts[cid] = counts.get(cid, 0) + 1
    return counts

def count_gt_instances_by_cat(gt_json_path: str) -> Dict[int, int]:
    coco_gt = COCO(gt_json_path)
    res: Dict[int, int] = {}
    for cid in coco_gt.getCatIds():
        ann_ids = coco_gt.getAnnIds(catIds=[cid])
        res[int(cid)] = len(ann_ids)
    return res

# ===================== JSON→라벨맵 렌더 + per-class 픽셀 정확도 ===================== #
def build_ann_index_by_image_id(anns: List[Dict[str,Any]]) -> Dict[str, List[Dict[str,Any]]]:
    idx: Dict[str, List[Dict[str,Any]]] = {}
    for a in anns:
        img_id = a.get("image_id")
        keys = set()
        # 기본 문자열 키
        keys.add(str(img_id))
        # 숫자 키가 가능하면 stem 키도 만들어 둠(파일명이 들어온 경우 대비)
        if isinstance(img_id, str):
            stem = os.path.splitext(os.path.basename(img_id))[0]
            keys.add(stem)
            if img_id.isdigit():
                keys.add(str(int(img_id)))
        elif isinstance(img_id, int):
            keys.add(str(img_id))
        for k in keys:
            idx.setdefault(k, []).append(a)
    return idx

def ann_to_mask(ann: Dict[str,Any], H: int, W: int) -> np.ndarray:
    seg = ann.get("segmentation", None)
    if seg is None:
        return np.zeros((H,W), dtype=np.uint8)

    if isinstance(seg, dict) and ("counts" in seg) and ("size" in seg):
        m = maskUtils.decode(seg)
        if m.ndim == 3:
            m = m[...,0]
        if m.shape != (H,W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        return (m > 0).astype(np.uint8)

    if isinstance(seg, list):
        if len(seg) == 0:
            return np.zeros((H,W), dtype=np.uint8)
        if all(isinstance(s, dict) and ("counts" in s) and ("size" in s) for s in seg):
            acc = np.zeros((H,W), dtype=np.uint8)
            for rle in seg:
                m = maskUtils.decode(rle)
                if m.ndim == 3:
                    m = m[...,0]
                if m.shape != (H,W):
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                acc |= (m > 0).astype(np.uint8)
            return acc
        rles = maskUtils.frPyObjects(seg, H, W)
        m = maskUtils.decode(rles)
        if m.ndim == 3:
            m = np.any(m, axis=2).astype(np.uint8)
        else:
            m = (m > 0).astype(np.uint8)
        return m

    return np.zeros((H,W), dtype=np.uint8)

def render_json_to_label_map_for_stem(stem: str, ann_index: Dict[str, List[Dict[str,Any]]], ref_size: Tuple[int,int]) -> np.ndarray:
    H, W = ref_size
    lab = np.zeros((H,W), dtype=np.int32)
    anns = ann_index.get(stem)
    if anns is None and stem.isdigit():
        anns = ann_index.get(str(int(stem)))
    if not anns:
        return lab

    for ann in anns:
        cat_id = int(ann.get("category_id", 0))
        if cat_id not in ID2BGR:
            continue
        m = ann_to_mask(ann, H, W)
        lab[m > 0] = cat_id
    return lab

def pixel_accuracy_per_class_from_json(json_path: str, label_dir: str) -> Dict[int, float]:
    """
    각 클래스별 픽셀 정확도: TP_c / GT_c_pixels
    (GT: LABEL_PATH에서 읽어 -1 시프트, Pred: json 렌더)
    """
    data = load_json(json_path)
    anns = data["annotations"] if isinstance(data, dict) and "annotations" in data else data
    ann_idx = build_ann_index_by_image_id(anns)

    stems = sorted(list_basenames(label_dir))
    if not stems:
        return {}

    # 누적용
    tp_by_c: Dict[int, int] = {cid: 0 for cid in EVAL_CLASS_IDS}
    gt_by_c: Dict[int, int] = {cid: 0 for cid in EVAL_CLASS_IDS}

    it = tqdm(stems, desc="Per-class pixel accuracy", unit="img") if USE_TQDM else stems
    for s in it:
        gp = find_with_any_ext(label_dir, s)
        gt_img = cv2.imread(gp, cv2.IMREAD_UNCHANGED)
        if gt_img is None:
            continue
        gt_lab = to_label_index_image(gt_img, is_gt=True)  # GT만 -1
        H, W = gt_lab.shape

        pr_lab = render_json_to_label_map_for_stem(s, ann_idx, (H, W))

        # ignore 제외 마스크
        vm = valid_mask(gt_lab)
        if not np.any(vm):
            continue

        # 클래스별 누적
        for cid in EVAL_CLASS_IDS:
            gt_mask_c = (gt_lab == cid) & vm
            if not np.any(gt_mask_c):
                continue
            pr_mask_c = (pr_lab == cid)
            tp_by_c[cid] += int(np.sum(gt_mask_c & pr_mask_c))
            gt_by_c[cid] += int(np.sum(gt_mask_c))

    acc_by_c: Dict[int, float] = {}
    for cid in EVAL_CLASS_IDS:
        acc_by_c[cid] = (tp_by_c[cid] / gt_by_c[cid]) if gt_by_c[cid] > 0 else float('nan')
    return acc_by_c

# ===================== 메인 ===================== #
def main():
    # 1) AP20 per class (eval-style)
    ap20_by_cat = ap20_per_class_evalstyle(COCO_GT_JSON, PRED_JSON)

    # 2) pixel accuracy per class
    pixacc_by_cat = pixel_accuracy_per_class_from_json(PRED_JSON, LABEL_DIR)

    # 3) #pred instances / #GT instances per class
    pred_counts = count_pred_instances_by_cat(PRED_JSON)
    gt_counts   = count_gt_instances_by_cat(COCO_GT_JSON)

    # 4) CSV 저장
    headers = ["class", "AP20 (long)", "pixel accuracy", "# pred instances", "# GT instances"]
    rows = []
    for cid in EVAL_CLASS_IDS:
        name = ID2NAME.get(cid, str(cid))
        ap20 = ap20_by_cat.get(cid, float('nan'))
        pix  = pixacc_by_cat.get(cid, float('nan'))
        pc   = pred_counts.get(cid, 0)
        gc   = gt_counts.get(cid, 0)

        rows.append([
            name,
            f"{ap20:.6f}" if np.isfinite(ap20) else "-",
            f"{pix:.6f}"  if np.isfinite(pix)  else "-",
            str(pc),
            str(gc),
        ])

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"[저장] {OUT_CSV}")
    # 콘솔 출력도 보기 좋게
    print("\t".join(headers))
    for r in rows:
        print("\t".join(r))

if __name__ == "__main__":
    main()
