import os
import glob
import json
import csv
import sys
from typing import List, Tuple, Dict, Any

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import config as cfg

# --- 설정 ---
LABEL_DIR = cfg.LABEL_PATH
PRED_DIR = cfg.PRED_PATH
COCO_GT_JSON = cfg.COCO_ANNO_PATH_GT_VER2
PRED_JSON = cfg.MERGED_JSON_PATH
OUT_DIR = os.path.join(cfg.RESULT_PATH, "Table")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "table_2.csv")

EXCLUDE_IDS = [8, 10, 11]
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
ID2BGR = {c['id']: (c['color'][2], c['color'][1], c['color'][0]) for c in METAINFO}
EVAL_CLASS_IDS = [c['id'] for c in METAINFO if c['id'] not in ([0] + EXCLUDE_IDS)]
ID2NAME = {c['id']: c['name'] for c in METAINFO}


# --- 유틸리티 ---
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f: return json.load(f)


def to_label_index_image(img: np.ndarray, is_gt: bool) -> np.ndarray:
    if img is None: return None
    h, w = img.shape[:2]
    lab = np.zeros((h, w), dtype=np.int32)
    if img.ndim == 2:
        lab = img.astype(np.int32)
    else:
        for cid, bgr in ID2BGR.items():
            mask = np.all(img == np.array(bgr, dtype=np.uint8), axis=-1)
            lab[mask] = int(cid)
    if is_gt:
        lab = lab - 1
        lab[lab < 0] = 0
    return lab


def ann_to_mask(ann: Dict[str, Any], H: int, W: int) -> np.ndarray:
    seg = ann.get("segmentation")
    if not seg: return np.zeros((H, W), dtype=np.uint8)
    if isinstance(seg, dict) and "counts" in seg:
        m = maskUtils.decode(seg)
        if m.ndim == 3: m = m[..., 0]
        return cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) if m.shape != (H,
                                                                                                      W) else m.astype(
            np.uint8)
    if isinstance(seg, list):
        if not seg: return np.zeros((H, W), dtype=np.uint8)
        rles = maskUtils.frPyObjects(seg, H, W)
        m = maskUtils.decode(rles)
        return np.any(m, axis=2).astype(np.uint8) if m.ndim == 3 else (m > 0).astype(np.uint8)
    return np.zeros((H, W), dtype=np.uint8)


def ap20_per_class_evalstyle(gt_json_path: str, pred_json_path: str) -> Dict[int, float]:
    gt_data = load_json(gt_json_path)
    gt_data['annotations'] = [a for a in gt_data.get('annotations', []) if a.get('category_id') not in EXCLUDE_IDS]
    for i, ann in enumerate(gt_data['annotations']):
        ann['id'] = ann.get('id', i + 1)
        ann['iscrowd'] = ann.get('iscrowd', 0)
        if 'area' not in ann: ann['area'] = float(maskUtils.area(ann['segmentation'])) if 'segmentation' in ann else 0.0

    coco_gt = COCO();
    coco_gt.dataset = gt_data;
    coco_gt.createIndex()
    dt_data = load_json(pred_json_path)
    if isinstance(dt_data, list): dt_data = [d for d in dt_data if d.get('category_id') not in EXCLUDE_IDS]
    coco_dt = coco_gt.loadRes(dt_data)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
    coco_eval.params.catIds = EVAL_CLASS_IDS
    coco_eval.params.iouThrs = np.array([0.20], dtype=np.float32)
    coco_eval.evaluate();
    coco_eval.accumulate()

    # [수정] maxDets=100 (마지막 인덱스) 사용
    p = coco_eval.eval['precision'][0, :, :, 0, -1]
    cat_ids = coco_eval.params.catIds
    res = {}
    for idx, cid in enumerate(cat_ids):
        p_slice = p[:, idx]
        valid_p = p_slice[p_slice > -1]
        res[int(cid)] = float(np.mean(valid_p)) if valid_p.size > 0 else 0.0
    return res


def pixel_accuracy_per_class_from_json(json_path: str, label_dir: str) -> Dict[int, float]:
    data = load_json(json_path)
    anns = data["annotations"] if isinstance(data, dict) else data
    ann_idx = {}
    for a in anns:
        if int(a.get("category_id", 0)) in EXCLUDE_IDS: continue
        ann_idx.setdefault(str(a.get("image_id")), []).append(a)
    stems = sorted([os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.lower().endswith(('.png', '.jpg'))])
    tp_by_c = {cid: 0 for cid in EVAL_CLASS_IDS}
    gt_by_c = {cid: 0 for cid in EVAL_CLASS_IDS}
    for s in tqdm(stems, desc="Pixel Acc"):
        gp = glob.glob(os.path.join(label_dir, s + ".*"))[0]
        gt_lab = to_label_index_image(cv2.imread(gp, cv2.IMREAD_UNCHANGED), True)
        if gt_lab is None: continue
        H, W = gt_lab.shape
        pr_lab = np.zeros((H, W), dtype=np.int32)
        cur_anns = ann_idx.get(s) or ann_idx.get(str(int(s)) if s.isdigit() else None)
        if cur_anns:
            for ann in cur_anns: pr_lab[ann_to_mask(ann, H, W) > 0] = int(ann.get("category_id", 0))
        for cid in EVAL_CLASS_IDS:
            gt_mask_c = (gt_lab == cid)
            if not np.any(gt_mask_c): continue
            tp_by_c[cid] += int(np.sum(gt_mask_c & (pr_lab == cid)))
            gt_by_c[cid] += int(np.sum(gt_mask_c))
    return {cid: (tp_by_c[cid] / gt_by_c[cid]) if gt_by_c[cid] > 0 else 0.0 for cid in EVAL_CLASS_IDS}


def main():
    ap20_by_cat = ap20_per_class_evalstyle(COCO_GT_JSON, PRED_JSON)
    pixacc_by_cat = pixel_accuracy_per_class_from_json(PRED_JSON, LABEL_DIR)
    gt_data = load_json(COCO_GT_JSON)
    pred_data = load_json(PRED_JSON)
    headers = ["class", "AP20", "pixel accuracy", "# pred", "# GT"]
    rows = []
    for cid in EVAL_CLASS_IDS:
        rows.append([ID2NAME.get(cid), f"{ap20_by_cat.get(cid, 0.0):.6f}", f"{pixacc_by_cat.get(cid, 0.0):.6f}",
                     str(sum(
                         1 for a in (pred_data if isinstance(pred_data, list) else pred_data.get('annotations', [])) if
                         int(a.get('category_id')) == cid)),
                     str(sum(1 for a in gt_data.get('annotations', []) if int(a.get('category_id')) == cid))])
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f);
        writer.writerow(headers);
        writer.writerows(rows)
    print("\n" + "=" * 100 + f"\n{'Table 2 Final Results':^100}\n" + "=" * 100)
    for r in [headers] + rows: print("\t".join([f"{str(cell):<20}" for cell in r]))


if __name__ == "__main__":
    main()