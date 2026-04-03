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
import pandas as pd
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

# --- 경로 및 설정 ---
LABEL_DIR = cfg.LABEL_PATH
PRED_DIR = cfg.PRED_PATH
COCO_GT_JSON = cfg.COCO_ANNO_PATH_GT_VER2
ORIGIN_JSON = cfg.ORIGIN_JSON_PATH
MERGED_JSON = cfg.MERGED_JSON_PATH

OUT_DIR = os.path.join(cfg.RESULT_PATH, "Table")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "table_1.csv")

USE_TQDM = True
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

# --- 유틸리티 ---
def list_basenames(folder: str):
    exts = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG")
    files = []
    for e in exts: files.extend(glob.glob(os.path.join(folder, f"*{e}")))
    return set(os.path.splitext(os.path.basename(p))[0] for p in files)

def find_with_any_ext(folder: str, stem: str):
    for e in (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"):
        p = os.path.join(folder, stem + e)
        if os.path.exists(p): return p
    return ""

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def to_label_index_image(img: np.ndarray, is_gt: bool) -> np.ndarray:
    if img is None: return None
    h, w = img.shape[:2]
    lab = np.zeros((h, w), dtype=np.int32)
    if img.ndim == 2: lab = img.astype(np.int32)
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
        return cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) if m.shape != (H, W) else m.astype(np.uint8)
    if isinstance(seg, list):
        if not seg: return np.zeros((H, W), dtype=np.uint8)
        rles = maskUtils.frPyObjects(seg, H, W)
        m = maskUtils.decode(rles)
        return np.any(m, axis=2).astype(np.uint8) if m.ndim == 3 else (m > 0).astype(np.uint8)
    return np.zeros((H, W), dtype=np.uint8)

def coco_ap_eval(gt_json: str, pred_json: str, ious=[0.10, 0.20, 0.50]):
    print(f"===== [coco_ap_eval] pred_json: {pred_json}")
    gt_data = load_json(gt_json)
    gt_data['annotations'] = [a for a in gt_data.get('annotations', []) if a.get('category_id') not in EXCLUDE_IDS]
    for i, ann in enumerate(gt_data['annotations']):
        ann['id'] = ann.get('id', i + 1)
        ann['iscrowd'] = ann.get('iscrowd', 0)
        if 'area' not in ann and 'segmentation' in ann:
            ann['area'] = float(maskUtils.area(ann['segmentation']) if isinstance(ann['segmentation'], dict) else 0.0)

    temp_gt = "temp_gt_eval_t1.json"
    with open(temp_gt, 'w') as f: json.dump(gt_data, f)
    coco_gt = COCO(temp_gt)
    dt_data = load_json(pred_json)
    if isinstance(dt_data, list): dt_data = [d for d in dt_data if d.get('category_id') not in EXCLUDE_IDS]
    coco_dt = coco_gt.loadRes(dt_data)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
    coco_eval.params.catIds = EVAL_CLASS_IDS
    coco_eval.params.iouThrs = np.array(ious, dtype=np.float32)
    coco_eval.evaluate(); coco_eval.accumulate()

    res = {"instances": str(len(coco_dt.getAnnIds()))}
    for i, iou in enumerate(ious):
        # [수정] maxDets=100 (마지막 인덱스)만 사용하여 평균 계산
        p = coco_eval.eval['precision'][i, :, :, 0, -1]
        p = p[p > -1]
        val = float(np.mean(p)) if p.size else 0.0
        res[f"AP{int(iou*100)}"] = val
    if os.path.exists(temp_gt): os.remove(temp_gt)
    print(f"===== [coco_ap_eval] res: {res}")
    return res

def pixel_accuracy_json_vs_dir(json_path: str, label_dir: str) -> Dict[str, str]:
    print(f"===== [pixel_accuracy_json_vs_dir] json_path: {json_path}, label_dir: {label_dir}")
    data = load_json(json_path)
    anns = data["annotations"] if isinstance(data, dict) else data
    ann_idx = {}
    for a in anns:
        if int(a.get("category_id", 0)) in EXCLUDE_IDS: continue
        ann_idx.setdefault(str(a.get("image_id")), []).append(a)
    stems = sorted(list_basenames(label_dir))
    correct, total = 0, 0
    for s in tqdm(stems, desc="JSON Pixel Acc"):
        gp = find_with_any_ext(label_dir, s)
        gt_lab = to_label_index_image(cv2.imread(gp, cv2.IMREAD_UNCHANGED), True)
        if gt_lab is None: continue
        H, W = gt_lab.shape
        pr_lab = np.zeros((H, W), dtype=np.int32)
        cur_anns = ann_idx.get(s) or ann_idx.get(str(int(s)) if s.isdigit() else None)
        if cur_anns:
            for ann in cur_anns: pr_lab[ann_to_mask(ann, H, W) > 0] = int(ann.get("category_id", 0))
        mask = (gt_lab != 0) & (~np.isin(gt_lab, EXCLUDE_IDS))
        if not np.any(mask): continue
        correct += int(np.sum((gt_lab == pr_lab) & mask))
        total += int(np.sum(mask))
    val = correct / total if total > 0 else 0
    res = {"pixel accuracy": val}
    print(f"===== [pixel_accuracy_json_vs_dir] res: {res}")
    return res

def main():
    print(f"Evaluating 8 classes: {EVAL_CLASS_IDS}")
    
    res_origin = {"algorithms": "linestrings"}
    res_origin.update(coco_ap_eval(COCO_GT_JSON, ORIGIN_JSON))
    res_origin.update(pixel_accuracy_json_vs_dir(ORIGIN_JSON, LABEL_DIR))
    
    res_merged = {"algorithms": "linestrings merged"}
    res_merged.update(coco_ap_eval(COCO_GT_JSON, MERGED_JSON))
    res_merged.update(pixel_accuracy_json_vs_dir(MERGED_JSON, LABEL_DIR))

    # headers = ["algorithms", "instances", "AP10", "AP20", "AP50", "pixel accuracy"]
    df = pd.DataFrame([res_origin, res_merged])
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\n{'Table 1 Final Results':^100}\n")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()