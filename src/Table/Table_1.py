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
LABEL_DIR = cfg.LABEL_PATH
PRED_DIR = cfg.PRED_PATH

COCO_GT_JSON = cfg.COCO_ANNO_PATH

ORIGIN_JSON = cfg.ORIGIN_JSON_PATH
ORIGIN_EXCEPTED_JSON = cfg.ORIGIN_EXCEPTED_JSON_PATH

MERGED_JSON = cfg.MERGED_JSON_PATH
MERGED_EXCEPTED_JSON = cfg.MERGED_EXCEPTED_JSON_PATH

OUT_DIR = os.path.join(cfg.RESULT_PATH, "Table")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "table_1.csv")

# ===== 옵션 =====
USE_TQDM = True
VIS_ENABLE = True
VIS_N_SAMPLES = 20
VIS_SAVE_DIR = os.path.join(OUT_DIR, "table1_visuals")
if VIS_ENABLE:
    os.makedirs(VIS_SAVE_DIR, exist_ok=True)

# ===================== 클래스 팔레트 ===================== #
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


# ===================== 유틸리티 함수 ===================== #
def list_basenames(folder: str, exts=(".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG")) -> set:
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, f"*{e}")))
    return set(os.path.splitext(os.path.basename(p))[0] for p in files)


def find_with_any_ext(folder: str, stem: str) -> str:
    for e in (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"):
        p = os.path.join(folder, stem + e)
        if os.path.exists(p): return p
    return ""


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_label_index_image(img: np.ndarray, *, is_gt: bool) -> np.ndarray:
    if img is None: return None
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


def label_index_to_color(lab: np.ndarray) -> np.ndarray:
    h, w = lab.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, bgr in ID2BGR.items():
        out[lab == cid] = bgr
    return out


def pixel_accuracy_from_dirs(gt_dir: str, pred_dir: str) -> float:
    stems = sorted(list(list_basenames(gt_dir) & list_basenames(pred_dir)))
    if not stems: return float("nan")
    correct, total = 0, 0
    it = tqdm(stems, desc="Segmentation pixel acc", unit="img") if USE_TQDM else stems
    for s in it:
        gp, pp = find_with_any_ext(gt_dir, s), find_with_any_ext(pred_dir, s)
        gt, pr = cv2.imread(gp, cv2.IMREAD_UNCHANGED), cv2.imread(pp, cv2.IMREAD_UNCHANGED)
        if gt is None or pr is None: continue
        gt_lab = to_label_index_image(gt, is_gt=True)
        pr_lab = to_label_index_image(pr, is_gt=False)
        if pr_lab.shape != gt_lab.shape:
            pr_lab = cv2.resize(pr_lab.astype(np.int32), (gt_lab.shape[1], gt_lab.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        ignore_mask = (gt_lab != 0)
        if not np.any(ignore_mask): continue
        correct += int(np.sum((gt_lab == pr_lab) & ignore_mask))
        total += int(np.sum(ignore_mask))
    return (correct / total) if total > 0 else float("nan")


def ann_to_mask(ann: Dict[str, Any], H: int, W: int) -> np.ndarray:
    seg = ann.get("segmentation", None)
    if seg is None: return np.zeros((H, W), dtype=np.uint8)
    if isinstance(seg, dict) and ("counts" in seg):
        m = maskUtils.decode(seg)
        if m.ndim == 3: m = m[..., 0]
        if m.shape != (H, W): m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        return (m > 0).astype(np.uint8)
    if isinstance(seg, list):
        if len(seg) == 0: return np.zeros((H, W), dtype=np.uint8)
        rles = maskUtils.frPyObjects(seg, H, W)
        m = maskUtils.decode(rles)
        if m.ndim == 3:
            m = np.any(m, axis=2).astype(np.uint8)
        else:
            m = (m > 0).astype(np.uint8)
        return m
    return np.zeros((H, W), dtype=np.uint8)


def pixel_accuracy_from_json_vs_labels(json_path: str, label_dir: str) -> float:
    data = load_json(json_path)
    anns = data["annotations"] if isinstance(data, dict) and "annotations" in data else data
    ann_idx = {}
    for a in anns:
        img_id = str(a.get("image_id"))
        ann_idx.setdefault(img_id, []).append(a)

    stems = sorted(list(list_basenames(label_dir)))
    if not stems: return float("nan")
    correct, total, vis_count = 0, 0, 0
    it = tqdm(stems, desc=f"JSON pixel acc ({os.path.basename(json_path)})", unit="img") if USE_TQDM else stems
    for s in it:
        gp = find_with_any_ext(label_dir, s)
        gt_img = cv2.imread(gp, cv2.IMREAD_UNCHANGED)
        if gt_img is None: continue
        gt_lab = to_label_index_image(gt_img, is_gt=True)
        H, W = gt_lab.shape
        pr_lab = np.zeros((H, W), dtype=np.int32)
        cur_anns = ann_idx.get(s) or (ann_idx.get(str(int(s))) if s.isdigit() else None)
        if cur_anns:
            for ann in cur_anns:
                cat_id = int(ann.get("category_id", 0))
                if cat_id in ID2BGR:
                    m = ann_to_mask(ann, H, W)
                    pr_lab[m > 0] = cat_id
        if VIS_ENABLE and vis_count < VIS_N_SAMPLES:
            vis = np.concatenate([label_index_to_color(gt_lab), label_index_to_color(pr_lab)], axis=1)
            cv2.imwrite(os.path.join(VIS_SAVE_DIR, f"{s}.png"), vis)
            vis_count += 1
        ignore_mask = (gt_lab != 0)
        if not np.any(ignore_mask): continue
        correct += int(np.sum((gt_lab == pr_lab) & ignore_mask))
        total += int(np.sum(ignore_mask))
    return (correct / total) if total > 0 else float("nan")


def _ap_at_iou_from_eval(coco_eval: COCOeval, iou: float) -> float:
    ths = coco_eval.params.iouThrs
    idx = np.where(np.isclose(ths, iou))[0]
    if idx.size == 0: return float('nan')
    p = coco_eval.eval['precision'][idx[0]]
    p = p[p > -1]
    return float(np.mean(p)) if p.size else float('nan')


# -------- [핵심 보정 부분: id, image_id, iscrowd, area 보정 및 인스턴스 카운트] -------- #
def coco_ap_set_evalstyle(gt_json: str, pred_json: str, ious=(0.10, 0.20, 0.50)) -> Tuple[Dict[str, float], int]:
    with open(gt_json, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    ref_h = gt_data['images'][0]['height'] if (gt_data.get('images') and 'height' in gt_data['images'][0]) else 1080
    ref_w = gt_data['images'][0]['width'] if (gt_data.get('images') and 'width' in gt_data['images'][0]) else 1920

    for i, img in enumerate(gt_data.get('images', [])):
        if 'id' not in img: img['id'] = i

    for i, ann in enumerate(gt_data.get('annotations', [])):
        if 'id' not in ann: ann['id'] = i
        if 'image_id' not in ann:
            ann['image_id'] = gt_data['images'][0]['id'] if gt_data.get('images') else 0
        if 'iscrowd' not in ann: ann['iscrowd'] = 0
        if 'area' not in ann:
            if 'segmentation' in ann:
                seg = ann['segmentation']
                try:
                    if isinstance(seg, list):
                        rles = maskUtils.frPyObjects(seg, ref_h, ref_w)
                        ann['area'] = float(np.sum(maskUtils.area(rles)))
                    else:
                        ann['area'] = float(maskUtils.area(seg))
                except:
                    ann['area'] = 0.0
            else:
                ann['area'] = 0.0

    temp_gt_path = gt_json.replace('.json', '_fixed_vfinal.json')
    with open(temp_gt_path, 'w', encoding='utf-8') as f:
        json.dump(gt_data, f)

    coco_gt = COCO(temp_gt_path)
    coco_dt = coco_gt.loadRes(pred_json)

    # [정확한 인스턴스 개수 추출] 실제 로드된 어노테이션 아이디의 개수
    num_instances = len(coco_dt.getAnnIds())

    ious_arr = np.array([float(x) for x in ious], dtype=np.float32)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
    coco_eval.params.iouThrs = ious_arr
    coco_eval.evaluate()
    coco_eval.accumulate()

    results = {f"{float(iou):.2f}": _ap_at_iou_from_eval(coco_eval, float(iou)) for iou in ious_arr}
    if os.path.exists(temp_gt_path): os.remove(temp_gt_path)

    return results, num_instances


def main():
    # 1. Segmentation Accuracy
    seg_pixel_acc = pixel_accuracy_from_dirs(LABEL_DIR, PRED_DIR)

    # 2. Original Linestrings
    # 개수를 함수에서 직접 받아옵니다.
    ap_all_lines, count_all = coco_ap_set_evalstyle(COCO_GT_JSON, ORIGIN_JSON)
    ap_long_lines, _ = coco_ap_set_evalstyle(COCO_GT_JSON, ORIGIN_EXCEPTED_JSON)
    pixel_acc_lines = pixel_accuracy_from_json_vs_labels(ORIGIN_JSON, LABEL_DIR)

    # 3. Merged Linestrings
    ap_all_merged, count_merged = coco_ap_set_evalstyle(COCO_GT_JSON, MERGED_JSON)
    ap_long_merged, _ = coco_ap_set_evalstyle(COCO_GT_JSON, MERGED_EXCEPTED_JSON)
    pixel_acc_merged = pixel_accuracy_from_json_vs_labels(MERGED_JSON, LABEL_DIR)

    # 4. Table Generation
    headers = ["", "instances", "AP10 (all)", "AP20 (all)", "AP50 (all)", "AP10 (long)", "AP20 (long)", "AP50 (long)",
               "pixel accuracy"]
    f_p = lambda x: f"{x:.6f}" if (x is not None and np.isfinite(x)) else "-"

    rows = [
        ["segmentation map", "-", "-", "-", "-", "-", "-", "-", f_p(seg_pixel_acc)],
        ["linestrings", str(count_all),
         f_p(ap_all_lines.get("0.10")), f_p(ap_all_lines.get("0.20")), f_p(ap_all_lines.get("0.50")),
         f_p(ap_long_lines.get("0.10")), f_p(ap_long_lines.get("0.20")), f_p(ap_long_lines.get("0.50")),
         f_p(pixel_acc_lines)],
        ["linestrings merged", str(count_merged),
         f_p(ap_all_merged.get("0.10")), f_p(ap_all_merged.get("0.20")), f_p(ap_all_merged.get("0.50")),
         f_p(ap_long_merged.get("0.10")), f_p(ap_long_merged.get("0.20")), f_p(ap_long_merged.get("0.50")),
         f_p(pixel_acc_merged)],
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print("\n" + "=" * 110)
    print(f"{'Table 1 Results (Instance Counts Verified)':^110}")
    print("=" * 110)
    for r in [headers] + rows:
        print("\t".join([f"{str(cell):<18}" for cell in r]))
    print("=" * 110)
    print(f"[저장 완료] {OUT_CSV}")


if __name__ == "__main__":
    main()