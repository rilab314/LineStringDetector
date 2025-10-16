# make_figure2.py
# 변경 사항 요약
# - 동일한 한 장의 이미지를 사용해 2x2 Figure 생성 (이미지별로 1개 파일 저장)
# - (1,1): RAW + GT의 guiding_line(8)만 오버레이(불투명 치환)
# - (1,2): RAW + GT의 safety_zone(10)만 오버레이(불투명 치환)
# - (2,1): PRED에서 guiding_line(8)만 남기고 나머지는 검정
# - (2,2): PRED에서 safety_zone(10)만 남기고 나머지는 검정
# - 포함 조건: GT와 PRED 모두에서 guiding_line, safety_zone 각각의 픽셀 수가 >= 20 이어야 함
#   (필요시 MIN_PIXELS 값을 조절하세요)
import os
import glob
import cv2
import numpy as np
from typing import List, Tuple, Dict
import config as cfg

# ===================== 경로/설정 ===================== #
RAW_DIR  = cfg.ORIGIN_PATH
GT_DIR   = cfg.GT_PATH
PRED_DIR = cfg.PRED_PATH

OUTPUT_DIR = os.path.join(cfg.RESULT_PATH, 'Figure', 'figure_2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 타일 간 여백과 배경색
GAP_PX   = 20
BG_COLOR = (255, 255, 255)  # BGR(흰색)

# 픽셀 임계치
MIN_PIXELS = 20
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
GUIDING_ID = 8
SAFETY_ID  = 10
GUIDING_BGR = ID2BGR[GUIDING_ID]
SAFETY_BGR  = ID2BGR[SAFETY_ID]

# ---------- 파일 매칭 유틸 ----------
def list_basenames(folder: str, exts=(".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG")) -> set:
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, f"*{e}")))
    return set(os.path.splitext(os.path.basename(p))[0] for p in files)

def get_common_stems(raw_dir: str, gt_dir: str, pred_dir: str) -> List[str]:
    raw_names  = list_basenames(raw_dir)
    gt_names   = list_basenames(gt_dir)
    pred_names = list_basenames(pred_dir)
    common = sorted(list(raw_names & gt_names & pred_names))
    print(f"[DEBUG] RAW ({raw_dir})  : {len(raw_names)}")
    print(f"[DEBUG] GT  ({gt_dir})   : {len(gt_names)}")
    print(f"[DEBUG] PRED({pred_dir}) : {len(pred_names)}")
    print(f"[DEBUG] 공통 basename 개수: {len(common)}")
    return common

def find_with_any_ext(folder: str, stem: str) -> str:
    for e in (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"):
        p = os.path.join(folder, stem + e)
        if os.path.exists(p):
            return p
    return ""

# ---------- 이미지/마스크 유틸 ----------
def resize_like(img: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    w, h = size_wh
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

def mask_from_color(img_bgr: np.ndarray, color_bgr: Tuple[int, int, int]) -> np.ndarray:
    """해당 색상과 완전히 일치하는 픽셀만 마스크(0/255)로 반환"""
    return (np.all(img_bgr == np.array(color_bgr, dtype=np.uint8), axis=-1)).astype(np.uint8) * 255

def count_pixels(mask_255: np.ndarray) -> int:
    """마스크(0/255)에서 255인 픽셀 수 반환"""
    return int(np.sum(mask_255 > 0))

def overlay_class_on_raw(raw_bgr: np.ndarray, gt_bgr: np.ndarray, class_bgr: Tuple[int,int,int]) -> np.ndarray:
    """raw 위에 GT의 특정 클래스만 불투명하게 오버레이"""
    H, W = raw_bgr.shape[:2]
    gt_bgr = resize_like(gt_bgr, (W, H))
    cls_mask = mask_from_color(gt_bgr, class_bgr)
    out = raw_bgr.copy()
    if np.any(cls_mask):
        out[cls_mask > 0] = class_bgr
    return out

def filter_pred_single_class(pred_bgr: np.ndarray, class_bgr: Tuple[int,int,int]) -> np.ndarray:
    """pred에서 특정 클래스만 남기고 나머지는 검정"""
    keep = np.all(pred_bgr == np.array(class_bgr, dtype=np.uint8), axis=-1)
    out = np.zeros_like(pred_bgr, dtype=np.uint8)
    out[keep] = class_bgr
    return out

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

def vstack_with_gap(images: List[np.ndarray], gap: int = GAP_PX, bg_color: Tuple[int,int,int] = BG_COLOR) -> np.ndarray:
    h, w = images[0].shape[:2]
    fixed = [cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA) if im.shape[:2] != (h, w) else im for im in images]
    canvas_h = h * len(fixed) + gap * (len(fixed) - 1)
    canvas = np.full((canvas_h, w, 3), bg_color, dtype=np.uint8)
    y = 0
    for im in fixed:
        canvas[y:y+h, 0:w] = im
        y += h + gap
    return canvas

# ---------- Figure 생성 (2x2, 동일 이미지로 구성) ----------
def build_figure2_per_image(raw_bgr: np.ndarray, gt_bgr: np.ndarray, pred_bgr: np.ndarray) -> np.ndarray:
    """ 하나의 이미지로 2x2 Figure 구성
        (1,1) RAW+GT (guiding), (1,2) RAW+GT (safety)
        (2,1) PRED (guiding only), (2,2) PRED (safety only)
    """
    H, W = raw_bgr.shape[:2]
    gt_bgr   = resize_like(gt_bgr,   (W, H))
    pred_bgr = resize_like(pred_bgr, (W, H))

    # 1행
    top_left  = overlay_class_on_raw(raw_bgr, gt_bgr,   GUIDING_BGR)
    top_right = overlay_class_on_raw(raw_bgr, gt_bgr,   SAFETY_BGR)

    # 2행
    bot_left  = filter_pred_single_class(pred_bgr, GUIDING_BGR)
    bot_right = filter_pred_single_class(pred_bgr, SAFETY_BGR)

    row1 = hstack_with_gap([top_left, top_right])
    row2 = hstack_with_gap([bot_left, bot_right])
    return vstack_with_gap([row1, row2])

# ---------- 포함 조건 검사 ----------
def passes_min_pixels(gt_bgr: np.ndarray, pred_bgr: np.ndarray, min_pixels: int = MIN_PIXELS) -> bool:
    """GT/PRED 모두에서 guiding, safety 각각 픽셀 수가 min_pixels 이상인지 검사"""
    gt_g = count_pixels(mask_from_color(gt_bgr,   GUIDING_BGR))
    gt_s = count_pixels(mask_from_color(gt_bgr,   SAFETY_BGR))
    pr_g = count_pixels(mask_from_color(pred_bgr, GUIDING_BGR))
    pr_s = count_pixels(mask_from_color(pred_bgr, SAFETY_BGR))

    ok = (gt_g >= min_pixels) and (gt_s >= min_pixels) and (pr_g >= min_pixels) and (pr_s >= min_pixels)
    return ok

# ---------- main ----------
def main():
    stems = get_common_stems(RAW_DIR, GT_DIR, PRED_DIR)
    if not stems:
        print("[경고] 공통 파일명이 없습니다.")
        return

    valid_count = 0
    saved_count = 0

    for stem in stems:
        rp = find_with_any_ext(RAW_DIR,  stem)
        gp = find_with_any_ext(GT_DIR,   stem)
        pp = find_with_any_ext(PRED_DIR, stem)
        if not rp or not gp or not pp:
            continue

        raw  = cv2.imread(rp)
        gt   = cv2.imread(gp)
        pred = cv2.imread(pp)
        if raw is None or gt is None or pred is None:
            continue

        # 포함 조건 검사 (GT/PRED 모두에서 두 클래스 각각 ≥ MIN_PIXELS)
        if not passes_min_pixels(gt, pred, MIN_PIXELS):
            continue

        valid_count += 1

        # Figure 생성 및 저장 (한 이미지 당 1개 파일)
        fig = build_figure2_per_image(raw, gt, pred)
        out_name = f"{stem}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, fig)
        saved_count += 1
        print(f"[저장] {out_path}")

    print(f"[INFO] 조건 충족 이미지 수: {valid_count}")
    print(f"[완료] 총 {saved_count}개 Figure 저장")

if __name__ == "__main__":
    main()
