# make_figure_1x4.py
# 목적:
# - 공통 basename을 갖는 이미지들에 대해 "1행 × 4열" 타일 이미지를 개별 생성
# - 4열: (1) RAW, (2) RAW + GT, (3) RAW + SEG(label), (4) RAW + PROCESS(JSON 복원)
# - 경로/설정은 config.py에서 불러옴
#     RAW_DIR   = cfg.ORIGIN_PATH
#     GT_DIR    = cfg.GT_PATH
#     LABEL_DIR = cfg.LABEL_PATH (또는 SEG_LABEL_PATH)
#     PROC_JSON = cfg.MERGED_JSON_PATH  # COCO 예측 포맷 호환
# - 결과 저장: cfg.RESULT_PATH/Figure/figure_1x4/{stem}.png

import os
import glob
import json
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from pycocotools import mask as maskUtils
import config as cfg

# ===================== 경로/설정 ===================== #
RAW_DIR    = cfg.ORIGIN_PATH
GT_DIR     = cfg.GT_PATH
# LABEL_PATH 또는 SEG_LABEL_PATH 둘 중 하나 기대
LABEL_DIR  = cfg.PRED_PATH

PROC_JSON  = cfg.MERGED_JSON_PATH  # process RLE/Polygon JSON (COCO 예측 포맷)

OUTPUT_DIR = os.path.join(cfg.RESULT_PATH, 'sample_image')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 타일 간 여백과 배경색
GAP_PX   = 20
BG_COLOR = (255, 255, 255)  # BGR(흰색)
# ==================================================== #

# METAINFO (RGB) → BGR 변환
METAINFO = [
    {'id': 0,  'name': 'ignore',                        'color': (0, 0, 0)},
    {'id': 1,  'name': 'center_line',                   'color': (77, 77, 255)},
    {'id': 2,  'name': 'u_turn_zone_line',              'color': (77, 178, 255)},
    {'id': 3,  'name': 'lane_line',                     'color': (77, 255, 77)},
    {'id': 4,  'name': 'bus_only_lane',                 'color': (255, 153, 77)},
    {'id': 5,  'name': 'edge_line',                     'color': (255, 77, 77)},
    {'id': 6,  'name': 'path_change_restriction_line',  'color': (178, 77, 255)},
    {'id': 7,  'name': 'no_parking_stopping_line',      'color': (77, 255, 178)},
    {'id': 8,  'name': 'guiding_line',                  'color': (255, 178, 77)},   # RGB
    {'id': 9,  'name': 'stop_line',                     'color': (77, 102, 255)},
    {'id': 10, 'name': 'safety_zone',                   'color': (255, 77, 128)},   # RGB
    {'id': 11, 'name': 'bicycle_lane',                  'color': (128, 255, 77)},
]
# RGB → BGR
ID2BGR: Dict[int, Tuple[int, int, int]] = {
    c['id']: (c['color'][2], c['color'][1], c['color'][0]) for c in METAINFO
}
BACKGROUND_BGR = ID2BGR.get(0, (0, 0, 0))

# ---------- 파일 매칭 유틸 ----------
def list_basenames(folder: str, exts=(".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG")) -> set:
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, f"*{e}")))
    return set(os.path.splitext(os.path.basename(p))[0] for p in files)

def get_common_stems(*folders: str) -> List[str]:
    sets = [list_basenames(fd) for fd in folders]
    common = sorted(list(set.intersection(*sets))) if sets else []
    print("".join([
        f"[DEBUG] {os.path.basename(fd):>12}({fd}) : {len(s)}\n"
        for fd, s in zip(folders, sets)
    ]), end="")
    print(f"[DEBUG] 공통 basename 개수: {len(common)}")
    return common

def find_with_any_ext(folder: str, stem: str) -> str:
    for e in (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"):
        p = os.path.join(folder, stem + e)
        if os.path.exists(p):
            return p
    return ""

# ---------- 기본 유틸 ----------
def resize_like(img: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    w, h = size_wh
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

def hstack_with_gap(images: List[np.ndarray], gap: int = GAP_PX, bg_color: Tuple[int,int,int] = BG_COLOR) -> np.ndarray:
    h, w = images[0].shape[:2]
    fixed = [cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA) if im.shape[:2] != (h, w) else im for im in images]
    canvas_w = w * len(fixed) + gap * (len(fixed) - 1)
    canvas = np.full((h, canvas_w, 3), bg_color, dtype=np.uint8)
    x = 0
    for im in fixed:
        canvas[0:h, x:x+w] = im
        x += w + gap
    return canvas

# ---------- 오버레이 ----------
def overlay_replace(raw_bgr: np.ndarray, overlay_bgr: np.ndarray) -> np.ndarray:
    """
    overlay_bgr에서 배경(0,0,0)이 아닌 픽셀을 raw 위에 불투명 치환.
    """
    H, W = raw_bgr.shape[:2]
    overlay_bgr = resize_like(overlay_bgr, (W, H))
    out = raw_bgr.copy()
    non_bg = ~np.all(overlay_bgr == np.array(BACKGROUND_BGR, dtype=np.uint8), axis=-1)
    out[non_bg] = overlay_bgr[non_bg]
    return out

def overlay_all_classes_on_raw_from_gt(raw_bgr: np.ndarray, gt_bgr: np.ndarray) -> np.ndarray:
    """
    GT 이미지가 팔레트(BGR)로 색칠되어 있다고 가정.
    배경(0,0,0)을 제외한 픽셀을 raw 위에 불투명 치환.
    """
    return overlay_replace(raw_bgr, gt_bgr)

# ---------- PROCESS JSON 처리 ----------
def load_process_json(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "annotations" in data:
        return data["annotations"]
    assert isinstance(data, list), "PROCESS JSON은 list[annotation] 형식이어야 합니다."
    return data

def build_proc_index_by_image_id(anns: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_img: Dict[str, List[Dict[str, Any]]] = {}
    for a in anns:
        img_id = str(a.get("image_id"))
        by_img.setdefault(img_id, []).append(a)
    return by_img

def ann_to_mask(ann: Dict[str, Any], H: int, W: int) -> np.ndarray:
    seg = ann.get("segmentation", None)
    if seg is None:
        return np.zeros((H, W), dtype=np.uint8)

    if isinstance(seg, dict) and ("counts" in seg) and ("size" in seg):
        m = maskUtils.decode(seg)
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        return (m > 0).astype(np.uint8)

    if isinstance(seg, list):
        if len(seg) == 0:
            return np.zeros((H, W), dtype=np.uint8)

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

        try:
            rles = maskUtils.frPyObjects(seg, H, W)
            m = maskUtils.decode(rles)
            if m.ndim == 3:
                m = np.any(m, axis=2).astype(np.uint8)
            else:
                m = (m > 0).astype(np.uint8)
            return m
        except Exception as e:
            print(f"[WARN] polygon decode 실패 (image_id={ann.get('image_id')}, cat={ann.get('category_id')}): {e}")
            return np.zeros((H, W), dtype=np.uint8)

    return np.zeros((H, W), dtype=np.uint8)

def render_process_color_map(ann_list: List[Dict[str, Any]], H: int, W: int) -> np.ndarray:
    if not ann_list:
        return None
    color_map = np.zeros((H, W, 3), dtype=np.uint8)
    for ann in ann_list:
        cat_id = int(ann.get("category_id", -1))
        color = ID2BGR.get(cat_id, None)
        if color is None:
            continue
        mask = ann_to_mask(ann, H, W)
        if mask is None or mask.size == 0:
            continue
        m = mask.astype(bool)
        color_map[m] = color
    return color_map  # (H,W,3) BGR

# ---------- LABEL → 컬러맵 ----------
def label_to_colormap_bgr(label_2d: np.ndarray) -> np.ndarray:
    """
    단일 채널 정수 라벨 이미지를 METAINFO에 따라 BGR 컬러맵으로 변환.
    """
    H, W = label_2d.shape[:2]
    color_map = np.zeros((H, W, 3), dtype=np.uint8)
    for cid, bgr in ID2BGR.items():
        m = (label_2d == cid)
        if np.any(m):
            color_map[m] = bgr
    return color_map

# ---------- main ----------
def main():
    # 공통 basename: RAW & GT & LABEL 기준
    stems = get_common_stems(RAW_DIR, GT_DIR, LABEL_DIR)
    if not stems:
        print("[경고] 공통 파일명이 없습니다.")
        return

    # PROCESS JSON 로딩 + image_id 인덱스
    proc_by_img = {}
    if PROC_JSON and os.path.exists(PROC_JSON):
        proc_anns = load_process_json(PROC_JSON)
        proc_by_img = build_proc_index_by_image_id(proc_anns)
        print(f"[DEBUG] PROCESS annotations: {len(proc_anns)}")
        print(f"[DEBUG] PROCESS image_id 개수: {len(proc_by_img)}")
    else:
        print("[INFO] PROCESS JSON 미지정 또는 없음 → raw+process는 raw로 대체")

    made = 0
    for stem in stems:
        rp = find_with_any_ext(RAW_DIR,   stem)
        gp = find_with_any_ext(GT_DIR,    stem)
        lp = find_with_any_ext(LABEL_DIR, stem)
        if not (rp and gp and lp):
            print(f"[WARN] 파일 누락: raw={bool(rp)} gt={bool(gp)} label={bool(lp)} stem={stem}")
            continue

        raw = cv2.imread(rp)
        gt  = cv2.imread(gp)
        if raw is None or gt is None:
            print(f"[WARN] 이미지 로드 실패 stem={stem}")
            continue

        # label은 정수 클래스 유지 위해 UNCHANGED로 읽기
        label = cv2.imread(lp, cv2.IMREAD_UNCHANGED)
        if label is None:
            print(f"[WARN] label 로드 실패 stem={stem}")
            continue

        # 여기 추가: 라벨이 1부터 시작한다면 -1 해서 0부터 시작하도록 보정
        label = label.astype(np.int32) - 1
        label[label < 0] = 0  # 음수 방지 → background는 0으로

        H, W = raw.shape[:2]

        # (1) RAW
        raw_bgr = raw

        # (2) RAW + GT (팔레트 GT를 그대로 오버레이)
        raw_plus_gt = overlay_all_classes_on_raw_from_gt(raw_bgr, gt)

        # (3) RAW + SEG (이미 색칠된 BGR seg 이미지를 raw 위에 오버레이, 검정=ignore는 제외)
        seg_bgr = cv2.imread(lp)  # 컬러로 읽기


        if seg_bgr is None:
            print(f"[WARN] seg 이미지 로드 실패 stem={stem}")
            raw_plus_seg = raw_bgr.copy()
        else:
            raw_plus_seg = overlay_replace(raw_bgr, seg_bgr)

        # (4) RAW + PROCESS (JSON 복원 컬러맵 → 오버레이)
        ann_list = proc_by_img.get(str(stem))
        if ann_list is None and str(stem).isdigit():
            ann_list = proc_by_img.get(str(int(stem)))
        if ann_list:
            proc_color = render_process_color_map(ann_list, H, W)
            raw_plus_proc = overlay_replace(raw_bgr, proc_color) if proc_color is not None else raw_bgr.copy()
        else:
            raw_plus_proc = raw_bgr.copy()

        # 1행 × 4열 타일 생성 및 저장
        row = hstack_with_gap([raw_bgr, raw_plus_gt, raw_plus_seg, raw_plus_proc], gap=GAP_PX, bg_color=BG_COLOR)
        out_path = os.path.join(OUTPUT_DIR, f"{stem}.png")
        cv2.imwrite(out_path, row)
        made += 1
        print(f"[저장] {out_path}")

    print(f"[완료] 총 {made}개 결과 저장")

if __name__ == "__main__":
    main()
