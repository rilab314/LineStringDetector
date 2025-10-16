import os
import cv2
import numpy as np

import config as cfg

# ================= 사용자 입력(직접 수정) =================
A_DIR = cfg.ORIGIN_PATH   # a 이미지 폴더
B_DIR = cfg.PRED_PATH   # b 이미지 폴더
C_DIR = os.path.join(cfg.WORK_PATH, 'results', 'figure_4', 'c')   # c 이미지 폴더
D_DIR = os.path.join(cfg.WORK_PATH, 'results', 'figure_4', 'd')   # d 이미지 폴더
OUT_DIR = os.path.join(cfg.WORK_PATH, 'results', 'figure_4', 'result')  # 결과 저장 폴더
os.makedirs(OUT_DIR, exist_ok=True)
# =========================================================

# ===== 파라미터(원하면 조정) =====
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
ROW_TARGET_HEIGHT = 256     # 각 이미지의 세로 높이(비율 유지)
GAP_H = 7                  # 가로 간격 (a|b|c|d 사이)
GAP_V = 7                  # 세로 간격 (각 row 사이)
MARGIN = 0                 # 바깥 여백 (상하좌우)
BG_COLOR = (255, 255, 255)  # 배경색 (BGR) - 논문용 흰 배경 권장
ROWS_PER_FIG = 4            # 한 장에 들어갈 row 수(4행)
# ===============================

def ensure_dir(path: str):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def load_image_as_bgr(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def resize_keep_aspect(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(
        img, (new_w, target_h),
        interpolation=cv2.INTER_AREA if target_h < h else cv2.INTER_CUBIC
    )

def list_by_stem(folder: str):
    """
    폴더 내 이미지 파일을 {stem: full_path}로 매핑.
    같은 stem이 여러 확장자일 경우 사전순으로 앞서는 파일 사용.
    """
    mapping = {}
    if not os.path.isdir(folder):
        return mapping
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue
        stem, ext = os.path.splitext(fname)
        if ext.lower() in IMG_EXTS:
            if stem not in mapping or os.path.basename(fpath).lower() < os.path.basename(mapping[stem]).lower():
                mapping[stem] = fpath
    return mapping

def hstack_with_gaps(img_list, gap_px=20, bg_color=(255,255,255), margin_left=0, margin_right=0):
    """
    동일한 높이의 이미지들을 흰 배경 위 가로 연결.
    (좌/우 마진을 추가로 둘 수 있음)
    """
    assert len(img_list) > 0
    h = img_list[0].shape[0]
    assert all(im.shape[0] == h for im in img_list), "가로 연결 전, 높이를 맞춰주세요."

    widths = [im.shape[1] for im in img_list]
    total_w = margin_left + sum(widths) + gap_px * (len(img_list) - 1) + margin_right

    canvas = np.full((h, total_w, 3), bg_color, dtype=np.uint8)
    x = margin_left
    for i, im in enumerate(img_list):
        canvas[:, x:x+im.shape[1]] = im
        x += im.shape[1]
        if i < len(img_list) - 1:
            x += gap_px
    return canvas

def vstack_with_gaps(rows, gap_px=30, bg_color=(255,255,255), margin_top=0, margin_bottom=0, margin_left=0, margin_right=0):
    """
    폭이 제각각인 row 이미지를 세로로 쌓는다.
    좌/우 폭을 가장 넓은 row 에 맞춰 확장(배경 채움) 후 붙임.
    상/하/좌/우 마진 적용.
    """
    assert len(rows) > 0
    heights = [r.shape[0] for r in rows]
    widths  = [r.shape[1] for r in rows]
    total_h = margin_top + sum(heights) + gap_px * (len(rows) - 1) + margin_bottom
    total_w = margin_left + max(widths) + margin_right

    canvas = np.full((total_h, total_w, 3), bg_color, dtype=np.uint8)
    y = margin_top
    for i, r in enumerate(rows):
        h, w = r.shape[:2]
        # 좌우 가운데 정렬
        x = margin_left + (max(widths) - w)//2
        canvas[y:y+h, x:x+w] = r
        y += h
        if i < len(rows) - 1:
            y += gap_px
    return canvas

def chunk_list(lst, n):
    """리스트를 길이 n씩 끊어 반환"""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def make_figure4(a_dir, b_dir, c_dir, d_dir, out_dir):
    ensure_dir(out_dir)

    a_map = list_by_stem(a_dir)
    b_map = list_by_stem(b_dir)
    c_map = list_by_stem(c_dir)
    d_map = list_by_stem(d_dir)

    # 공통 stem만 사용
    common = sorted(set(a_map) & set(b_map) & set(c_map) & set(d_map))
    if not common:
        raise RuntimeError("공통 파일명이 없습니다. a/b/c/d 폴더에 같은 이름의 파일이 있는지 확인하세요.")

    print(f"[INFO] 공통 파일명 개수: {len(common)}")
    page_idx = 1

    # 4장씩 끊어서(= 4 rows per figure) figure_4_xxx.png 생성
    for stems in chunk_list(common, ROWS_PER_FIG):
        rows = []
        for stem in stems:
            a_img = load_image_as_bgr(a_map[stem])
            b_img = load_image_as_bgr(b_map[stem])
            c_img = load_image_as_bgr(c_map[stem])
            d_img = load_image_as_bgr(d_map[stem])

            if any(im is None for im in [a_img, b_img, c_img, d_img]):
                print(f"[WARN] 읽기 실패로 스킵: {stem}")
                continue

            # 각 row의 이미지 높이를 동일하게 맞춤
            a_r = resize_keep_aspect(a_img, ROW_TARGET_HEIGHT)
            b_r = resize_keep_aspect(b_img, ROW_TARGET_HEIGHT)
            c_r = resize_keep_aspect(c_img, ROW_TARGET_HEIGHT)
            d_r = resize_keep_aspect(d_img, ROW_TARGET_HEIGHT)

            row_img = hstack_with_gaps([a_r, b_r, c_r, d_r],
                                       gap_px=GAP_H,
                                       bg_color=BG_COLOR)
            rows.append(row_img)

        if not rows:
            print(f"[WARN] 페이지에 들어갈 유효 row가 없어 건너뜀 (page {page_idx})")
            continue

        # 행들을 세로로 쌓아 4x4 figure 생성
        fig = vstack_with_gaps(
            rows,
            gap_px=GAP_V,
            bg_color=BG_COLOR,
            margin_top=MARGIN, margin_bottom=MARGIN,
            margin_left=MARGIN, margin_right=MARGIN
        )

        out_path = os.path.join(out_dir, f"figure_4_{page_idx:03d}.png")
        ok = cv2.imwrite(out_path, fig)
        if ok:
            print(f"[OK] 저장: {out_path}")
        else:
            print(f"[ERROR] 저장 실패: {out_path}")
        page_idx += 1

    print("[DONE] figure_4 생성 완료.")

if __name__ == "__main__":
    make_figure4(A_DIR, B_DIR, C_DIR, D_DIR, OUT_DIR)
