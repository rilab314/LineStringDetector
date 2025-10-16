# make_figure5.py
# 목적:
# - 공통 basename을 갖는 이미지들 중 앞에서부터 5개씩 묶어 5×3 Figure 생성
# - 각 행(row)은 서로 다른 이미지 1장으로 구성 (총 5행)
# - 각 행의 3열(col):
#     (1) RAW
#     (2) RAW + GT(모든 클래스 색상 오버레이)
#     (3) RAW + PROCESS(모든 클래스 색상 오버레이)  ← process는 JSON(RLE/Polygon 혼재 가능)에서 복원
# - 경로/설정은 config.py에서 불러옴
#     RAW_DIR  = cfg.ORIGIN_PATH
#     GT_DIR   = cfg.GT_PATH
#     PROCESS_JSON_PATH = cfg.PROCESS_JSON_PATH
# - 결과 저장: cfg.RESULT_PATH/Figure/figure_5/*.png

import os
import glob
import json
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from pycocotools import mask as maskUtils
import config as cfg

# ===================== 경로/설정 ===================== #
RAW_DIR   = cfg.ORIGIN_PATH
GT_DIR    = cfg.GT_PATH
PROC_JSON = cfg.MERGED_JSON_PATH  # process RLE/Polygon JSON (COCO 예측 포맷)

OUTPUT_DIR = os.path.join(cfg.RESULT_PATH, 'Figure', 'figure_5')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 타일 간 여백과 배경색
GAP_PX   = 20
BG_COLOR = (255, 255, 255)  # BGR(흰색)

# 5행씩 묶음
ROWS_PER_FIG = 5
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
BACKGROUND_BGR = ID2BGR.get(0, (0,0,0))

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

# ---------- GT 오버레이 (모든 클래스) ----------
def overlay_all_classes_on_raw_from_gt(raw_bgr: np.ndarray, gt_bgr: np.ndarray) -> np.ndarray:
    """
    GT 이미지가 팔레트(BGR)로 색칠되어 있다고 가정.
    배경(0,0,0)을 제외한 픽셀을 raw 위에 불투명 치환.
    """
    H, W = raw_bgr.shape[:2]
    gt_bgr = resize_like(gt_bgr, (W, H))
    out = raw_bgr.copy()
    non_bg = ~np.all(gt_bgr == np.array(BACKGROUND_BGR, dtype=np.uint8), axis=-1)
    out[non_bg] = gt_bgr[non_bg]
    return out

# ---------- PROCESS JSON 처리 ----------
def load_process_json(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "annotations" in data:
        # COCO-style dict로 감싼 형태
        return data["annotations"]
    assert isinstance(data, list), "PROCESS JSON은 list[annotation] 형식이어야 합니다."
    return data

def build_proc_index_by_image_id(anns: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    image_id가 str 또는 int일 수 있으므로 모두 str로 통일하여 인덱싱.
    """
    by_img: Dict[str, List[Dict[str, Any]]] = {}
    for a in anns:
        img_id = str(a.get("image_id"))
        by_img.setdefault(img_id, []).append(a)
    return by_img

def ann_to_mask(ann: Dict[str, Any], H: int, W: int) -> np.ndarray:
    """
    annotation → 바이너리 마스크(0/1, shape=(H,W))
    segmentation 타입(RLE dict / RLE list / polygon list)을 모두 처리.
    """
    seg = ann.get("segmentation", None)
    if seg is None:
        return np.zeros((H, W), dtype=np.uint8)

    # (A) RLE dict
    if isinstance(seg, dict) and ("counts" in seg) and ("size" in seg):
        rle = seg
        # decode
        m = maskUtils.decode(rle)
        if m.ndim == 3:
            m = m[..., 0]
        # 크기가 다르면 리사이즈(최근접)
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        return (m > 0).astype(np.uint8)

    # (B) list 인 경우: polygon list 또는 RLE list
    if isinstance(seg, list):
        # polygon list (각 원소가 [x1,y1, x2,y2, ...]) 또는 dict RLE들의 리스트일 수 있음
        # dict RLE가 섞여 있으면 각각 decode해서 OR, polygon이면 frPyObjects로 래스터라이즈
        if len(seg) == 0:
            return np.zeros((H, W), dtype=np.uint8)

        # 모든 원소가 dict(counts,size)면 RLE list로 간주
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

        # 그 외는 polygon list로 처리
        try:
            rles = maskUtils.frPyObjects(seg, H, W)  # H,W 주의: (height,width)
            m = maskUtils.decode(rles)               # (H,W,n) 또는 (H,W)
            if m.ndim == 3:
                m = np.any(m, axis=2).astype(np.uint8)
            else:
                m = (m > 0).astype(np.uint8)
            return m
        except Exception as e:
            print(f"[WARN] polygon decode 실패 (image_id={ann.get('image_id')}, cat={ann.get('category_id')}): {e}")
            return np.zeros((H, W), dtype=np.uint8)

    # 알 수 없는 타입
    return np.zeros((H, W), dtype=np.uint8)

def render_process_color_map(ann_list: List[Dict[str, Any]], H: int, W: int) -> np.ndarray:
    """
    동일 image_id의 process annotations(list) → 클래스 컬러 맵(BGR) 생성
    겹칠 경우, 뒤에 오는 ann이 앞을 덮어씀(간단한 z-order).
    """
    if not ann_list:
        return None

    color_map = np.zeros((H, W, 3), dtype=np.uint8)

    for ann in ann_list:
        cat_id = int(ann.get("category_id", -1))
        color = ID2BGR.get(cat_id, None)
        if color is None:
            # 정의되지 않은 클래스는 건너뜀
            continue

        mask = ann_to_mask(ann, H, W)  # 0/1
        if mask is None or mask.size == 0:
            continue

        m = mask.astype(bool)
        color_map[m] = color

    return color_map  # (H,W,3) BGR

def overlay_on_raw(raw_bgr: np.ndarray, overlay_bgr: np.ndarray) -> np.ndarray:
    """
    overlay_bgr에서 배경(0,0,0)이 아닌 픽셀을 raw 위에 불투명 치환.
    """
    H, W = raw_bgr.shape[:2]
    overlay_bgr = resize_like(overlay_bgr, (W, H))
    out = raw_bgr.copy()
    non_bg = ~np.all(overlay_bgr == np.array(BACKGROUND_BGR, dtype=np.uint8), axis=-1)
    out[non_bg] = overlay_bgr[non_bg]
    return out

# ---------- Figure 생성 (5×3) ----------
def build_figure5_block(rows_triplets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    rows_triplets: 길이 5 리스트, 각 원소는 (raw, raw+gt, raw+proc)
    각 행을 가로로 이어붙이고, 5행을 세로로 이어붙여 최종 Figure 생성.
    """
    assert len(rows_triplets) == ROWS_PER_FIG
    row_imgs = [hstack_with_gap([raw, raw_gt, raw_proc]) for (raw, raw_gt, raw_proc) in rows_triplets]
    figure = vstack_with_gap(row_imgs)
    return figure

# ---------- main ----------
def main():
    # 공통 basename: RAW & GT 기준
    stems = get_common_stems(RAW_DIR, GT_DIR)
    if not stems:
        print("[경고] 공통 파일명이 없습니다.")
        return

    # PROCESS JSON 로딩 + image_id 인덱스
    proc_anns = load_process_json(PROC_JSON)
    proc_by_img = build_proc_index_by_image_id(proc_anns)
    print(f"[DEBUG] PROCESS annotations: {len(proc_anns)}")
    print(f"[DEBUG] PROCESS image_id 개수: {len(proc_by_img)}")

    total_figs = 0
    for i in range(0, len(stems), ROWS_PER_FIG):
        group = stems[i:i+ROWS_PER_FIG]
        if len(group) < ROWS_PER_FIG:
            print(f"[SKIP] 마지막 그룹 {len(group)}개 (<{ROWS_PER_FIG})")
            break

        rows: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        ok_group = True

        for stem in group:
            rp = find_with_any_ext(RAW_DIR, stem)
            gp = find_with_any_ext(GT_DIR,  stem)
            if not rp or not gp:
                print(f"[WARN] 파일 누락: raw={bool(rp)} gt={bool(gp)} stem={stem}")
                ok_group = False
                break

            raw = cv2.imread(rp)
            gt  = cv2.imread(gp)
            if raw is None or gt is None:
                print(f"[WARN] 이미지 로드 실패 stem={stem}")
                ok_group = False
                break

            H, W = raw.shape[:2]

            # (1) RAW
            raw_bgr = raw

            # (2) RAW + GT(모든 클래스)
            raw_plus_gt = overlay_all_classes_on_raw_from_gt(raw_bgr, gt)

            # (3) RAW + PROCESS(모든 클래스)
            ann_list = proc_by_img.get(str(stem))
            if ann_list is None and stem.isdigit():
                ann_list = proc_by_img.get(str(int(stem)))

            if ann_list:
                proc_color = render_process_color_map(ann_list, H, W)
                if proc_color is not None:
                    raw_plus_proc = overlay_on_raw(raw_bgr, proc_color)
                else:
                    raw_plus_proc = raw_bgr.copy()
            else:
                raw_plus_proc = raw_bgr.copy()

            rows.append((raw_bgr, raw_plus_gt, raw_plus_proc))

        if not ok_group or len(rows) != ROWS_PER_FIG:
            print(f"[SKIP] 그룹 구성 실패 (rows={len(rows)})")
            continue

        fig = build_figure5_block(rows)
        out_name = "_".join(group) + ".png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, fig)
        total_figs += 1
        print(f"[저장] {out_path}")

    print(f"[완료] 총 {total_figs}개 Figure 저장")

if __name__ == "__main__":
    main()
