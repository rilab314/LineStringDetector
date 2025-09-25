# render_json_to_bgr.py
import os
import json
import cv2
import numpy as np
from typing import Dict, Tuple, Any, List, Union
from pycocotools import mask as maskUtils
import config as cfg

# ===================== 사용자 설정 (파이참에서 직접 수정) ===================== #
COCO_GT_JSON    = cfg.COCO_ANNO_PATH     # GT COCO annotation json
PRED_JSON       = cfg.MERGED_JSON_PATH   # 예측 json (COCO results 형식 또는 annotations 포함 dict)
OUT_GT_DIR      = os.path.join(cfg.RESULT_PATH, 'GT')
OUT_PRED_DIR = os.path.join(cfg.RESULT_PATH, 'PRED')

# ===================== METAINFO / 팔레트 ===================== #
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
# BGR 팔레트 (OpenCV용)
ID2BGR: Dict[int, Tuple[int, int, int]] = {
    c['id']: (c['color'][2], c['color'][1], c['color'][0]) for c in METAINFO
}

# ===================== 유틸 ===================== #
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def label_index_to_bgr(lab: np.ndarray) -> np.ndarray:
    """(H,W) 인덱스 라벨맵을 METAINFO의 BGR 팔레트로 변환"""
    h, w = lab.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, bgr in ID2BGR.items():
        out[lab == cid] = bgr
    return out

def ann_to_mask(ann: Dict[str, Any], H: int, W: int) -> np.ndarray:
    """
    COCO segmentation to binary mask (H,W), supports polygon / RLE / RLE-list.
    """
    seg = ann.get("segmentation", None)
    if seg is None:
        return np.zeros((H, W), dtype=np.uint8)

    # 압축 RLE(dict)
    if isinstance(seg, dict) and ("counts" in seg) and ("size" in seg):
        m = maskUtils.decode(seg)
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        return (m > 0).astype(np.uint8)

    # 리스트: 폴리곤 또는 RLE 리스트
    if isinstance(seg, list):
        if len(seg) == 0:
            return np.zeros((H, W), dtype=np.uint8)

        # RLE 리스트
        if all(isinstance(s, dict) and ("counts" in s) and ("size" in s) for s in seg):
            acc = np.zeros((H, W), dtype=np.uint8)
            for rle in seg:
                m = maskUtils.decode(rle)
                if m.ndim == 3:
                    m = m[..., 0]
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                acc |= (m > 0).astype(np.uint8)
            return acc

        # 폴리곤
        rles = maskUtils.frPyObjects(seg, H, W)
        m = maskUtils.decode(rles)
        if m.ndim == 3:
            m = np.any(m, axis=2).astype(np.uint8)
        else:
            m = (m > 0).astype(np.uint8)
        return m

    return np.zeros((H, W), dtype=np.uint8)

def build_image_meta_from_gt(gt: Dict[str, Any]) -> Dict[Union[int, str], Dict[str, Any]]:
    """
    GT JSON의 images 배열로부터 이미지 메타(폭/높이/파일명 등) 구성.
    반환 키: image_id(int)와 file_name(stem) 둘 다 접근 가능하도록 확장.
    """
    meta: Dict[Union[int, str], Dict[str, Any]] = {}
    images = gt.get("images", [])
    for im in images:
        img_id = im.get("id")
        file_name = im.get("file_name", "")
        w = int(im.get("width", 0))
        h = int(im.get("height", 0))
        stem = os.path.splitext(os.path.basename(file_name))[0]

        entry = {"id": img_id, "file_name": file_name, "width": w, "height": h, "stem": stem}
        # 키: 정수 image_id
        if img_id is not None:
            meta[img_id] = entry
        # 키: 문자열 image_id (혹시 문자열로 오는 경우 대비)
        if isinstance(img_id, (int, np.integer)):
            meta[str(int(img_id))] = entry
        elif isinstance(img_id, str):
            meta[img_id] = entry
        # 키: stem
        meta[stem] = entry
    return meta

def build_ann_index(anns: List[Dict[str, Any]]) -> Dict[Union[int, str], List[Dict[str, Any]]]:
    """
    image_id / image_id(str) / stem 로 접근 가능한 어노테이션 인덱스.
    """
    idx: Dict[Union[int, str], List[Dict[str, Any]]] = {}
    for a in anns:
        img_id = a.get("image_id")
        keys = set()

        # 기본 키들
        keys.add(img_id)
        keys.add(str(img_id))

        # stem 키도 확보(파일경로/파일명 형태의 image_id를 대비)
        if isinstance(img_id, str):
            stem = os.path.splitext(os.path.basename(img_id))[0]
            keys.add(stem)

        # 정규화
        norm_keys = []
        for k in keys:
            if isinstance(k, (int, np.integer)) or isinstance(k, str):
                norm_keys.append(k)

        for k in norm_keys:
            idx.setdefault(k, []).append(a)
    return idx

def collect_annotations(data: Any) -> List[Dict[str, Any]]:
    """pred json이 dict(annotations 포함)일 수도 있고 list(results)일 수도 있음."""
    if isinstance(data, dict):
        if "annotations" in data:
            return data["annotations"]
        if "results" in data:
            return data["results"]
    if isinstance(data, list):
        return data
    return []

def render_one_label_index(anns_for_img: List[Dict[str, Any]], H: int, W: int) -> np.ndarray:
    """
    해당 이미지의 모든 어노테이션을 라벨 인덱스맵(H,W)으로 렌더.
    후 오는 인스턴스가 앞을 덮어쓸 수 있음(일반적인 합성).
    """
    lab = np.zeros((H, W), dtype=np.int32)
    if not anns_for_img:
        return lab
    for ann in anns_for_img:
        cid = int(ann.get("category_id", 0))
        if cid not in ID2BGR:
            continue
        m = ann_to_mask(ann, H, W)
        lab[m > 0] = cid
    return lab

def save_bgr(lab: np.ndarray, out_path: str):
    bgr = label_index_to_bgr(lab)
    cv2.imwrite(out_path, bgr)

# ===================== 메인 로직 ===================== #
def render_json_using_gt_size(gt_json_path: str, target_json_path: str, out_dir: str, suffix: str):
    """
    GT의 이미지 메타(폭/높이)를 참조해 target_json(=GT 또는 Pred)를 BGR로 렌더하여 저장.
    파일명: <stem>_{suffix}.png
    """
    ensure_dir(out_dir)

    gt = load_json(gt_json_path)
    tgt = load_json(target_json_path)

    img_meta = build_image_meta_from_gt(gt)
    tgt_anns = collect_annotations(tgt)
    ann_idx = build_ann_index(tgt_anns)

    # GT의 images를 기준으로 반복
    images = gt.get("images", [])
    for im in images:
        img_id = im.get("id")
        file_name = im.get("file_name", "")
        w = int(im.get("width", 0))
        h = int(im.get("height", 0))
        stem = os.path.splitext(os.path.basename(file_name))[0]

        # 해당 이미지의 anns를 image_id / str(image_id) / stem 순으로 탐색
        anns_for_img = (
            ann_idx.get(img_id)
            or ann_idx.get(str(img_id))
            or ann_idx.get(stem)
            or []
        )

        lab = render_one_label_index(anns_for_img, h, w)
        out_path = os.path.join(out_dir, f"{stem}_{suffix}.png")
        save_bgr(lab, out_path)

    print(f"[완료] 저장 경로: {out_dir}")

if __name__ == "__main__":
    # GT JSON을 BGR로 렌더
    render_json_using_gt_size(
        gt_json_path=COCO_GT_JSON,
        target_json_path=COCO_GT_JSON,
        out_dir=OUT_GT_DIR,
        suffix="gt"
    )

    # Pred JSON을 BGR로 렌더
    render_json_using_gt_size(
        gt_json_path=COCO_GT_JSON,
        target_json_path=PRED_JSON,
        out_dir=OUT_PRED_DIR,
        suffix="pred"
    )
