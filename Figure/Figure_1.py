# split_and_stitch.py
# 파이참에서 그냥 실행하면 됩니다. (터미널 인자 불필요)
import os
import glob
import cv2
import numpy as np
from typing import Dict, List, Tuple
import config as cfg

# ===================== 하드코딩 설정 영역 ===================== #
# LineStringDetector.save_images 로 저장된 "큰 합성 이미지"들이 있는 폴더
INPUT_DIR = os.path.join(cfg.RESULT_PATH, 'result')   # 예) r"D:\work\lane\results"

# 결과 저장 폴더 (자동 생성)
OUTPUT_DIR = os.path.join(cfg.RESULT_PATH, 'Figure', 'Figure_1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 붙일 패널 선택 (아래 키 중에서 골라 순서대로 나열)
# 가능한 키: 'src_img', 'anno_img', 'pred_img', 'raw_line', 'merged_line', 'long_line'
SELECT_PANELS = ['src_img', 'pred_img', 'merged_line']   # 예) 필요한 조합으로 수정

# 가로('h')로 이어붙일지, 세로('v')로 이어붙일지
DIRECTION = 'h'

# 이미지 사이 간격 (px) — 흰색
GAP = 20

# 저장 파일명 접미사
FNAME_SUFFIX = "_stitched"

# 그리드 레이아웃 (ImageShow(columns=3), 총 6장이라면 rows=2)
GRID_COLS = 3
GRID_ROWS = 2

# ==== 라벨(패널 이름) 영역 지우기 옵션 ==== #
ERASE_LABEL = True                 # True면 라벨(텍스트) 영역을 흰색으로 덮음
ERASE_SCOPE = 'selected'           # 'selected' = 선택된 타일만, 'all' = 모든 타일
LABEL_POS = 'top'                  # 라벨 위치: 'top' 또는 'bottom'
LABEL_BAND_PX = 28                 # 지울 높이(px) — 필요에 따라 30~60 정도로 조정
LABEL_BG_COLOR = (255, 255, 255)   # 덮을 색 (흰색)
# ============================================================ #

# 합성 이미지 내부 타일 이름(이미지 저장 순서와 일치해야 함)
# images_to_save = {'src_img', 'anno_img', 'pred_img', 'raw_line', 'merged_line', 'long_line'}
PANEL_KEYS = ['src_img', 'anno_img', 'pred_img',
              'raw_line', 'merged_line', 'long_line']


def split_grid(img: np.ndarray, cols: int, rows: int) -> List[np.ndarray]:
    """그리드 합성 이미지를 cols x rows 타일로 균등 분할."""
    H, W = img.shape[:2]
    tile_w = W // cols
    tile_h = H // rows

    tiles = []
    for r in range(rows):
        for c in range(cols):
            x0 = c * tile_w
            y0 = r * tile_h
            x1 = (c + 1) * tile_w
            y1 = (r + 1) * tile_h
            tiles.append(img[y0:y1, x0:x1].copy())
    return tiles


def tiles_to_named_dict(tiles: List[np.ndarray], names: List[str]) -> Dict[str, np.ndarray]:
    """타일 리스트를 이름과 매핑."""
    n = min(len(tiles), len(names))
    return {names[i]: tiles[i] for i in range(n)}


def wipe_label_band(img: np.ndarray, pos: str = 'top', band: int = 40,
                    bg_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """타일 이미지의 상단/하단 문자열 영역을 흰색으로 덮어 지움."""
    h, w = img.shape[:2]
    band = max(0, min(band, h))
    if band == 0:
        return img
    out = img.copy()
    if pos.lower() == 'top':
        out[0:band, 0:w] = bg_color
    else:
        out[h-band:h, 0:w] = bg_color
    return out


def stitch_images(images: List[np.ndarray], direction: str = 'h', gap: int = 16,
                  bg_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """여러 이미지를 흰색 간격을 두고 가로나 세로로 붙임 (모두 같은 크기 가정)."""
    if not images:
        raise ValueError("붙일 이미지가 없습니다.")

    h0, w0 = images[0].shape[:2]
    # 혹시 크기 다르면 보정
    for i, im in enumerate(images):
        if im.shape[:2] != (h0, w0):
            images[i] = cv2.resize(im, (w0, h0), interpolation=cv2.INTER_AREA)

    if direction.lower() == 'h':
        total_w = w0 * len(images) + gap * (len(images) - 1)
        canvas = np.full((h0, total_w, 3), bg_color, dtype=np.uint8)
        x = 0
        for im in images:
            canvas[0:h0, x:x+w0] = im
            x += w0 + gap
    else:
        total_h = h0 * len(images) + gap * (len(images) - 1)
        canvas = np.full((total_h, w0, 3), bg_color, dtype=np.uint8)
        y = 0
        for im in images:
            canvas[y:y+h0, 0:w0] = im
            y += h0 + gap
    return canvas


def process_folder(input_dir: str, output_dir: str, select_names: List[str],
                   cols: int, rows: int, direction: str, gap: int,
                   suffix: str):
    os.makedirs(output_dir, exist_ok=True)

    # results 폴더 내 PNG/JPG 전부 처리 (너의 save_images는 .png 유지)
    exts = ('*.png', '*.jpg', '*.jpeg')
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    files.sort()

    if not files:
        print(f"[경고] 입력 폴더에 이미지가 없습니다: {input_dir}")
        return

    print(f"[정보] 총 {len(files)}개 파일 처리 시작")
    for idx, fp in enumerate(files, 1):
        img = cv2.imread(fp)
        if img is None:
            print(f"[건너뜀] 읽기 실패: {fp}")
            continue

        # 큰 합성 이미지를 그리드로 분할
        tiles = split_grid(img, cols=cols, rows=rows)
        named = tiles_to_named_dict(tiles, PANEL_KEYS)

        # (옵션) 모든 타일에 라벨 영역 지우기
        if ERASE_LABEL and ERASE_SCOPE == 'all':
            for k in list(named.keys()):
                named[k] = wipe_label_band(named[k], pos=LABEL_POS, band=LABEL_BAND_PX, bg_color=LABEL_BG_COLOR)

        # 선택 패널만 추출 (순서 유지)
        picked = []
        missing = []
        for name in select_names:
            if name in named:
                tile = named[name]
                # (옵션) 선택된 타일에만 라벨 영역 지우기
                if ERASE_LABEL and ERASE_SCOPE == 'selected':
                    tile = wipe_label_band(tile, pos=LABEL_POS, band=LABEL_BAND_PX, bg_color=LABEL_BG_COLOR)
                picked.append(tile)
            else:
                missing.append(name)

        if missing:
            print(f"[주의] '{os.path.basename(fp)}'에서 누락된 패널: {missing} (이 파일은 가능한 패널만 결합)")
        if not picked:
            print(f"[건너뜀] 결합할 패널이 없습니다: {fp}")
            continue

        stitched = stitch_images(picked, direction=direction, gap=gap)

        # 저장 파일명 구성
        base = os.path.splitext(os.path.basename(fp))[0]
        out_name = f"{base}.png"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, stitched)
        print(f"[{idx}/{len(files)}] 저장 완료: {out_path}")


def main():
    # 하드코딩된 설정 사용
    process_folder(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        select_names=SELECT_PANELS,
        cols=GRID_COLS,
        rows=GRID_ROWS,
        direction=DIRECTION,
        gap=GAP,
        suffix=FNAME_SUFFIX
    )


if __name__ == "__main__":
    main()
