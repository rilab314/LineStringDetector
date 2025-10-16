import cv2
import numpy as np
import os

import config as cfg

# ============== 폴더/설정 ==============
# 입력 폴더 경로
INFER_DIR = cfg.PRED_PATH  # inference 컬러 결과 이미지 폴더 (METAINFO 색 사용)
B_DIR     = os.path.join(cfg.WORK_PATH, 'results', 'figure_3', 'b')
C_DIR     = os.path.join(cfg.WORK_PATH, 'results', 'figure_3', 'c')
D_DIR     = os.path.join(cfg.WORK_PATH, 'results', 'figure_3', 'd')

# 출력 폴더 경로
A_DIR      = os.path.join(cfg.WORK_PATH, 'results', 'figure_3', 'modified_a')
FIGURE_DIR = os.path.join(cfg.WORK_PATH, 'results', 'figure_3', 'result')

# center_line 색상 (METAINFO id=1): BGR 가정
CENTER_LINE_BGR = (77, 77, 255)

# 채널별 허용 오차(압축 등으로 색이 조금 달라졌을 때 포착)
TOL = 5

# 이어붙일 때 설정
TARGET_HEIGHT = 512        # 네 이미지의 공통 높이 (비율 유지 리사이즈)
GAP_PX        = 20         # 이미지 사이 간격(픽셀)
BG_COLOR      = (255, 255, 255)  # 흰 배경 (BGR)

# 마스크 결합 모드: "auto" = 더 많이 잡히는 쪽 선택, "union" = BGR/RGB OR 결합
MODE = "auto"

# 허용 확장자
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
# =====================================


def load_image_as_bgr(path: str) -> np.ndarray:
    """
    이미지를 BGR 3채널로 로드.
    RGBA(4채널)면 알파 제거, 그레이면 BGR로 변환.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        pass
    else:
        return None
    return img


def two_masks(img_bgr, center_bgr=(77,77,255), tol=5):
    """
    BGR / RGB-as-BGR 두 경우의 마스크 모두 계산.
    """
    # BGR 기준
    lo1 = np.clip(np.array([center_bgr[0]-tol, center_bgr[1]-tol, center_bgr[2]-tol]), 0, 255).astype(np.uint8)
    hi1 = np.clip(np.array([center_bgr[0]+tol, center_bgr[1]+tol, center_bgr[2]+tol]), 0, 255).astype(np.uint8)
    mask_bgr = cv2.inRange(img_bgr, lo1, hi1) > 0

    # RGB 저장물을 BGR로 읽은 경우(채널 뒤집힘)
    alt = (center_bgr[2], center_bgr[1], center_bgr[0])
    lo2 = np.clip(np.array([alt[0]-tol, alt[1]-tol, alt[2]-tol]), 0, 255).astype(np.uint8)
    hi2 = np.clip(np.array([alt[0]+tol, alt[1]+tol, alt[2]+tol]), 0, 255).astype(np.uint8)
    mask_rgb_as_bgr = cv2.inRange(img_bgr, lo2, hi2) > 0

    return mask_bgr, mask_rgb_as_bgr


def extract_center_line_mixed(img_bgr,
                              center_bgr=(77,77,255),
                              tol=5,
                              mode="auto",
                              bg_color=(0,0,0)):
    """
    a 이미지 추출 알고리즘:
    mode="auto": 두 마스크 중 히트가 더 큰 쪽 선택
    mode="union": 두 마스크를 OR로 합침
    배경색 지정(bg_color). a는 검정 배경을 사용.
    """
    mask_bgr, mask_rgb = two_masks(img_bgr, center_bgr=center_bgr, tol=tol)
    hits_bgr = int(mask_bgr.sum())
    hits_rgb = int(mask_rgb.sum())

    if mode == "union":
        chosen_mask = mask_bgr | mask_rgb
        chosen_tag = "UNION"
    else:
        if hits_bgr >= hits_rgb:
            chosen_mask = mask_bgr
            chosen_tag = "BGR"
        else:
            chosen_mask = mask_rgb
            chosen_tag = "RGB_as_BGR"

    out = np.zeros_like(img_bgr, dtype=np.uint8)
    if bg_color != (0,0,0):
        out[:] = bg_color
    out[chosen_mask] = img_bgr[chosen_mask]
    return out, chosen_tag, hits_bgr, hits_rgb


def resize_keep_aspect(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, target_h),
                      interpolation=cv2.INTER_AREA if target_h < h else cv2.INTER_CUBIC)


def hstack_with_gaps(img_list, gap_px=20, bg_color=(255,255,255)):
    """
    동일한 높이의 이미지들을 흰 배경 캔버스 위에 가로로 나란히 붙임.
    """
    assert len(img_list) > 0
    heights = [im.shape[0] for im in img_list]
    widths  = [im.shape[1] for im in img_list]
    h = heights[0]
    assert all(ht == h for ht in heights), "모든 이미지의 높이가 같아야 합니다. (사전에 맞춰주세요)"

    total_w = sum(widths) + gap_px * (len(img_list) - 1)
    canvas = np.full((h, total_w, 3), bg_color, dtype=np.uint8)

    x = 0
    for i, im in enumerate(img_list):
        canvas[:, x:x+im.shape[1]] = im
        x += im.shape[1]
        if i < len(img_list) - 1:
            x += gap_px
    return canvas


def list_images_by_stem(folder: str):
    """
    폴더 내 이미지 파일을 stem(확장자 제외 파일명) -> 전체경로 로 매핑.
    같은 stem에 여러 확장자가 있을 경우, 알파벳 순으로 첫 번째를 사용.
    """
    mapping = {}
    if not os.path.isdir(folder):
        return mapping
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue
        root, ext = os.path.splitext(fname)
        if ext.lower() in IMG_EXTS:
            if root not in mapping or os.path.basename(fpath).lower() < os.path.basename(mapping[root]).lower():
                mapping[root] = fpath
    return mapping


def ensure_dir(d):
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def main():
    # 0) 출력 폴더 준비
    ensure_dir(A_DIR)
    ensure_dir(FIGURE_DIR)

    # 1) 폴더별 파일 매핑
    infer_map = list_images_by_stem(INFER_DIR)
    b_map     = list_images_by_stem(B_DIR)
    c_map     = list_images_by_stem(C_DIR)
    d_map     = list_images_by_stem(D_DIR)

    # 2) 공통 파일명(stem) 교집합만 처리
    common_stems = sorted(set(infer_map) & set(b_map) & set(c_map) & set(d_map))
    if not common_stems:
        raise RuntimeError("공통 파일명이 없습니다. 폴더 내 파일명이 서로 일치하는지 확인하세요.")

    skipped = {
        'infer_only': sorted(set(infer_map) - set(common_stems)),
        'b_only':     sorted(set(b_map) - set(common_stems)),
        'c_only':     sorted(set(c_map) - set(common_stems)),
        'd_only':     sorted(set(d_map) - set(common_stems)),
    }
    print(f"[INFO] 처리 예정 개수: {len(common_stems)}")
    for k, v in skipped.items():
        if v:
            print(f"[WARN] {k} 에만 있는 파일({len(v)}개): {v[:5]}{' ...' if len(v) > 5 else ''}")

    # 3) 루프 처리
    for stem in common_stems:
        infer_path = infer_map[stem]
        b_path     = b_map[stem]
        c_path     = c_map[stem]
        d_path     = d_map[stem]

        infer_img = load_image_as_bgr(infer_path)
        b_img     = load_image_as_bgr(b_path)
        c_img     = load_image_as_bgr(c_path)
        d_img     = load_image_as_bgr(d_path)

        if infer_img is None or b_img is None or c_img is None or d_img is None:
            print(f"[ERROR] 이미지 로드 실패. 건너뜀: {stem}")
            continue

        # a) center_line만 추출 (검정 배경, 혼재 대응)
        a_img, chosen, h_bgr, h_rgb = extract_center_line_mixed(
            infer_img,
            center_bgr=CENTER_LINE_BGR,
            tol=TOL,
            mode=MODE,
            bg_color=(0,0,0)  # a는 검정 배경
        )
        print(f"[{stem}] a-extract chosen={chosen} hits(BGR/RGBasBGR)={h_bgr}/{h_rgb}")

        # b) 세로 높이 맞춰 리사이즈
        a_r = resize_keep_aspect(a_img, TARGET_HEIGHT)
        b_r = resize_keep_aspect(b_img, TARGET_HEIGHT)
        c_r = resize_keep_aspect(c_img, TARGET_HEIGHT)
        d_r = resize_keep_aspect(d_img, TARGET_HEIGHT)

        # c) 가로 이어붙이기(흰 간격 포함)
        figure = hstack_with_gaps([a_r, b_r, c_r, d_r], gap_px=GAP_PX, bg_color=BG_COLOR)

        # d) 저장 (PNG 권장)
        a_out_path  = os.path.join(A_DIR, f"{stem}.png")
        fig_out_path = os.path.join(FIGURE_DIR, f"{stem}.png")

        ok_a = cv2.imwrite(a_out_path, a_img)
        ok_f = cv2.imwrite(fig_out_path, figure)

        if not ok_a or not ok_f:
            print(f"[ERROR] 저장 실패: {stem}")
        else:
            print(f"[OK] a 저장: {a_out_path} / figure 저장: {fig_out_path}")

    print("[DONE] 전체 처리 완료.")


if __name__ == "__main__":
    # 허용 확장자 전역 선언(함수에서 참조)
    IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    main()
