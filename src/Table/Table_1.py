import os
import glob
import json
import csv
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

LABEL_DIR   = cfg.LABEL_PATH
PRED_DIR    = cfg.PRED_PATH

COCO_GT_JSON                = cfg.COCO_ANNO_PATH

ORIGIN_JSON                 = cfg.ORIGIN_JSON_PATH
ORIGIN_EXCEPTED_JSON        = cfg.ORIGIN_EXCEPTED_JSON_PATH

MERGED_JSON                 = cfg.MERGED_JSON_PATH
MERGED_EXCEPTED_JSON        = cfg.MERGED_EXCEPTED_JSON_PATH

OUT_DIR = os.path.join(cfg.RESULT_PATH, "Table")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "table_1.csv")

# ===== 옵션 =====
USE_TQDM       = True                 # tqdm 진행률 사용
VIS_ENABLE     = True                 # JSON→pred 시각화 저장
VIS_N_SAMPLES  = 20                   # 저장할 샘플 개수 (앞에서부터)
VIS_SAVE_DIR   = os.path.join(OUT_DIR, "table1_visuals")
if VIS_ENABLE:
    os.makedirs(VIS_SAVE_DIR, exist_ok=True)

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

IGNORE_IDS = {0}     # pixel acc에서 무시할 라벨

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
        raise ValueError(f"Unsupported image shape for label conversion: {img.shape}")

    if is_gt:
        lab = lab - 1
        lab[lab < 0] = 0
    return lab

def label_index_to_color(lab: np.ndarray) -> np.ndarray:
    h, w = lab.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, bgr in ID2BGR.items():
        out[lab == cid] = bgr
    return out

def valid_mask(gt_lab: np.ndarray) -> np.ndarray:
    ignore_set = IGNORE_IDS if isinstance(IGNORE_IDS, set) else {IGNORE_IDS}
    vm = np.ones_like(gt_lab, dtype=bool)
    for ig in ignore_set:
        vm &= (gt_lab != ig)
    return vm

def pixel_accuracy_from_dirs(gt_dir: str, pred_dir: str) -> float:
    stems = sorted(list(list_basenames(gt_dir) & list_basenames(pred_dir)))
    if not stems:
        return float("nan")

    correct = 0
    total   = 0
    it = tqdm(stems, desc="Segmentation pixel acc", unit="img") if USE_TQDM else stems
    for s in it:
        gp = find_with_any_ext(gt_dir, s)
        pp = find_with_any_ext(pred_dir, s)
        gt = cv2.imread(gp, cv2.IMREAD_UNCHANGED)
        pr = cv2.imread(pp, cv2.IMREAD_UNCHANGED)
        if gt is None or pr is None:
            continue

        gt_lab = to_label_index_image(gt, is_gt=True)     # GT만 -1 시프트
        pr_lab = to_label_index_image(pr, is_gt=False)    # Pred는 시프트 없음

        H, W = gt_lab.shape
        if pr_lab.shape != (H, W):
            pr_lab = cv2.resize(pr_lab.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)

        vm = valid_mask(gt_lab)
        if not np.any(vm):
            continue
        match = (gt_lab == pr_lab) & vm
        correct += int(np.sum(match))
        total   += int(np.sum(vm))

    return (correct / total) if total > 0 else float("nan")

def build_ann_index_by_image_id(anns: List[Dict[str,Any]]) -> Dict[str, List[Dict[str,Any]]]:
    idx: Dict[str, List[Dict[str,Any]]] = {}
    for a in anns:
        img_id = str(a.get("image_id"))
        idx.setdefault(img_id, []).append(a)
    return idx

def ann_to_mask(ann: Dict[str,Any], H: int, W: int) -> np.ndarray:
    seg = ann.get("segmentation", None)
    if seg is None:
        return np.zeros((H,W), dtype=np.uint8)

    if isinstance(seg, dict) and ("counts" in seg) and ("size" in seg):  # RLE dict
        m = maskUtils.decode(seg)
        if m.ndim == 3:
            m = m[...,0]
        if m.shape != (H,W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        return (m > 0).astype(np.uint8)

    if isinstance(seg, list):
        if len(seg) == 0:
            return np.zeros((H,W), dtype=np.uint8)
        if all(isinstance(s, dict) and ("counts" in s) and ("size" in s) for s in seg):  # RLE list
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
    if anns is None:
        if stem.isdigit():
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

def pixel_accuracy_from_json_vs_labels(json_path: str, label_dir: str) -> float:
    data = load_json(json_path)
    anns = data["annotations"] if isinstance(data, dict) and "annotations" in data else data
    assert isinstance(anns, list), "Prediction JSON must be list[annotation] or {'annotations': [...]}"

    ann_idx = build_ann_index_by_image_id(anns)

    stems = sorted(list_basenames(label_dir))
    if not stems:
        return float("nan")

    correct = 0
    total   = 0
    vis_count = 0

    it = tqdm(stems, desc=f"JSON pixel acc ({os.path.basename(json_path)})", unit="img") if USE_TQDM else stems
    for s in it:
        gp = find_with_any_ext(label_dir, s)
        gt_img = cv2.imread(gp, cv2.IMREAD_UNCHANGED)
        if gt_img is None:
            continue
        gt_lab = to_label_index_image(gt_img, is_gt=True)   # GT만 -1 시프트
        H, W = gt_lab.shape

        pr_lab = render_json_to_label_map_for_stem(s, ann_idx, (H, W))  # pred JSON은 시프트 없음

        if VIS_ENABLE and vis_count < VIS_N_SAMPLES:
            gt_color = label_index_to_color(gt_lab)
            pr_color = label_index_to_color(pr_lab)
            vis = np.concatenate([gt_color, pr_color], axis=1)
            save_path = os.path.join(VIS_SAVE_DIR, f"{s}.png")
            cv2.imwrite(save_path, vis)
            vis_count += 1

        vm = valid_mask(gt_lab)
        if not np.any(vm):
            continue
        match = (gt_lab == pr_lab) & vm
        correct += int(np.sum(match))
        total   += int(np.sum(vm))

    return (correct / total) if total > 0 else float("nan")

def _ap_at_iou_from_eval(coco_eval: COCOeval, iou: float) -> float:
    ths = coco_eval.params.iouThrs  # [T]
    idx = np.where(np.isclose(ths, iou))[0]
    if idx.size == 0:
        return float('nan')
    p = coco_eval.eval['precision'][idx[0]]  # [R, K, A, M]
    p = p[p > -1]
    return float(np.mean(p)) if p.size else float('nan')

def coco_ap_set_evalstyle(gt_json: str, pred_json: str, ious=(0.10, 0.20, 0.50)) -> Dict[str, float]:
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(pred_json)

    ious_arr = np.array([float(x) for x in ious], dtype=np.float32)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
    coco_eval.params.iouThrs = ious_arr

    coco_eval.evaluate()
    coco_eval.accumulate()

    results = {}
    for iou in ious_arr:
        key = f"{float(iou):.2f}"
        results[key] = _ap_at_iou_from_eval(coco_eval, float(iou))
    return results

def main():
    seg_pixel_acc = pixel_accuracy_from_dirs(LABEL_DIR, PRED_DIR)

    origin_json_data = load_json(ORIGIN_JSON)
    origin_anns = origin_json_data["annotations"] if isinstance(origin_json_data, dict) and "annotations" in origin_json_data else origin_json_data
    instances_lines = len(origin_anns)

    ap_all_lines  = coco_ap_set_evalstyle(COCO_GT_JSON, ORIGIN_JSON,          ious=(0.10, 0.20, 0.50))
    ap_long_lines = coco_ap_set_evalstyle(COCO_GT_JSON, ORIGIN_EXCEPTED_JSON, ious=(0.10, 0.20, 0.50))

    pixel_acc_lines = pixel_accuracy_from_json_vs_labels(ORIGIN_JSON, LABEL_DIR)

    merged_json_data = load_json(MERGED_JSON)
    merged_anns = merged_json_data["annotations"] if isinstance(merged_json_data, dict) and "annotations" in merged_json_data else merged_json_data
    instances_merged = len(merged_anns)

    ap_all_merged  = coco_ap_set_evalstyle(COCO_GT_JSON, MERGED_JSON,          ious=(0.10, 0.20, 0.50))
    ap_long_merged = coco_ap_set_evalstyle(COCO_GT_JSON, MERGED_EXCEPTED_JSON, ious=(0.10, 0.20, 0.50))

    pixel_acc_merged = pixel_accuracy_from_json_vs_labels(MERGED_JSON, LABEL_DIR)

    headers = ["", "instances",
               "AP10 (all)", "AP20 (all)", "AP50 (all)",
               "AP10 (long)", "AP20 (long)", "AP50 (long)",
               "pixel accuracy"]

    def f(x): return f"{x:.6f}" if (x is not None and np.isfinite(x)) else "-"

    rows = [
        ["segmentation map", "-", "-", "-", "-", "-", "-", "-", f(seg_pixel_acc)],
        ["linestrings",
         str(instances_lines),
         f(ap_all_lines.get("0.10", float('nan'))),
         f(ap_all_lines.get("0.20", float('nan'))),
         f(ap_all_lines.get("0.50", float('nan'))),
         f(ap_long_lines.get("0.10", float('nan'))),
         f(ap_long_lines.get("0.20", float('nan'))),
         f(ap_long_lines.get("0.50", float('nan'))),
         f(pixel_acc_lines)
        ],
        ["linestrings merged",
         str(instances_merged),
         f(ap_all_merged.get("0.10", float('nan'))),
         f(ap_all_merged.get("0.20", float('nan'))),
         f(ap_all_merged.get("0.50", float('nan'))),
         f(ap_long_merged.get("0.10", float('nan'))),
         f(ap_long_merged.get("0.20", float('nan'))),
         f(ap_long_merged.get("0.50", float('nan'))),
         f(pixel_acc_merged)
        ],
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"[저장] {OUT_CSV}")
    if VIS_ENABLE:
        print(f"[참고] JSON→Pred 시각화 저장 경로: {VIS_SAVE_DIR}")

    for r in [headers] + rows:
        print("\t".join(map(str, r)))

if __name__ == "__main__":
    main()
