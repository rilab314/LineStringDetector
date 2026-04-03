import os
import json
import csv
import cv2
import glob
import numpy as np
from tqdm import tqdm

# 기존 모듈 임포트
import config as cfg
from lane_detector import LineStringDetector

# 평가 스크립트 임포트
try:
    from Table import Table_1 as t1
    from Table import Table_2 as t2
except ImportError:
    import Table_1 as t1
    import Table_2 as t2


def run_experiments():
    # 1. 파라미터 조합 설정
    sample_strides = [5, 10, 15]
    extend_lens = [20, 30, 40]
    thicknesses = [3, 5]

    # [수정] 테스트를 위해 10장만 실행하려면 이 값을 10으로 설정, 전체를 돌리려면 None으로 설정
    DEBUG_LIMIT = None

    base_result_path = cfg.RESULT_PATH

    # 2. 전체 실험 횟수 계산 (총 18회)
    total_runs = len(thicknesses) * len(sample_strides) * len(extend_lens)
    print(f"Total Experiments: {total_runs}")
    if DEBUG_LIMIT:
        print(f"!!! DEBUG MODE: Processing only {DEBUG_LIMIT} images per experiment !!!")

    # 전체 실험 진행률 (Outer tqdm)
    pbar = tqdm(total=total_runs, desc="Total Progress", position=0)

    for t in thicknesses:
        for s in sample_strides:
            for e in extend_lens:
                # ---------------------------------------------------------
                # A. 실험별 경로 설정
                # ---------------------------------------------------------
                exp_path = os.path.join(
                    base_result_path,
                    f"thickness={t}",
                    f"sample_stride={s}",
                    f"extend_len={e}"
                )

                result_dir = os.path.join(exp_path, 'result')
                vis_dir = os.path.join(result_dir, 'visuals')

                os.makedirs(result_dir, exist_ok=True)
                os.makedirs(vis_dir, exist_ok=True)

                # ---------------------------------------------------------
                # B. Detector 설정 및 실행
                # ---------------------------------------------------------
                # detector = LineStringDetector(cfg.DATA_PATH)
                # detector.thickness = t
                # detector.sample_stride = s
                # detector.extend_len = e

                # 현재 파라미터 정보를 설명으로 생성
                current_desc = f"[Run] T={t}/S={s}/E={e}"

                # limit과 desc를 전달
                # process_experiment(detector, cfg.DATA_PATH, result_dir, vis_dir,
                #                    limit=DEBUG_LIMIT, desc=current_desc)

                # ---------------------------------------------------------
                # C. 평가 및 테이블 생성
                # ---------------------------------------------------------
                # [수정] 평가 중 에러가 나더라도 전체 실험이 멈추지 않도록 try-except 처리 가능
                try:
                    generate_tables(exp_path, result_dir)
                except Exception as e:
                    print(f"\n[Error] Failed to generate tables for {current_desc}: {e}")

                pbar.update(1)

    pbar.close()
    print("\nAll experiments completed.")


def process_experiment(detector, src_path, result_dir, vis_dir, limit=None, desc="Processing"):
    """
    Args:
        limit (int): 테스트를 위해 처리할 이미지 수를 제한 (None이면 전체)
        desc (str): tqdm 진행바에 표시할 설명
    """
    # Validation 이미지 목록 가져오기
    img_files = sorted(glob.glob(os.path.join(src_path, 'images', 'validation', '*.png')))

    if limit is not None:
        img_files = img_files[:limit]

    origin_json = []
    origin_exc_json = []
    merged_json = []
    merged_exc_json = []

    # 내부 tqdm 적용
    for file_name in tqdm(img_files, desc=desc, leave=False, position=1):
        # 이미지 읽기
        image, pred_img, anno_img = detector._read_image(file_name)
        detector._img_shape = image.shape[:2]
        detector._id_count = detector.id_offset

        # 1. 라인 추출
        raw_lines, raw_img = detector.extract_lines(pred_img, file_name)

        # 2. 짧은 라인 제거 (Raw)
        raw_exc_lines, _ = detector._except_short_lines(raw_lines)

        # 3. 라인 병합 (Merge)
        m_lines, m_img = detector.merge_lines(raw_lines, 0)
        m_lines, m_img = detector.merge_lines(m_lines, 1)

        # 4. 짧은 라인 제거 (Merged)
        m_exc_lines, m_exc_img = detector._except_short_lines(m_lines)

        # 5. 시각화 저장
        images_to_save = {
            'src_img': image, 'anno_img': anno_img, 'pred_img': pred_img,
            'raw_line': raw_img, 'merged_line': m_img, 'long_line': m_exc_img
        }

        detector._imshow_save.show_imgs(images_to_save)
        combined_img = detector._imshow_save.update_whole_image()

        base_name = os.path.basename(file_name)
        save_path = os.path.join(vis_dir, base_name)

        # 이미지가 유효한 경우에만 저장
        if combined_img is not None and hasattr(combined_img, 'size') and combined_img.size > 0:
            cv2.imwrite(save_path, combined_img)
        else:
            pass

        # 6. JSON 데이터 축적
        img_id = base_name.replace('.png', '')
        origin_json = detector.accumulate_preds(raw_lines, img_id, origin_json)
        origin_exc_json = detector.accumulate_preds(raw_exc_lines, img_id, origin_exc_json)
        merged_json = detector.accumulate_preds(m_lines, img_id, merged_json)
        merged_exc_json = detector.accumulate_preds(m_exc_lines, img_id, merged_exc_json)

    # JSON 파일 저장
    save_json(result_dir, 'coco_pred_instances_origin.json', origin_json)
    save_json(result_dir, 'coco_pred_instances_origin_excepted.json', origin_exc_json)
    save_json(result_dir, 'coco_pred_instances_merged.json', merged_json)
    save_json(result_dir, 'coco_pred_instances_excepted.json', merged_exc_json)


def save_json(folder, filename, data):
    with open(os.path.join(folder, filename), 'w', encoding='utf-8') as f:
        json.dump(data, f)


def generate_tables(exp_path, result_dir):
    gt_json = cfg.COCO_ANNO_PATH_GT_VER2
    label_dir = cfg.LABEL_PATH
    pred_dir_static = cfg.PRED_PATH

    f_origin = os.path.join(result_dir, "coco_pred_instances_origin.json")
    f_origin_exc = os.path.join(result_dir, "coco_pred_instances_origin_excepted.json")
    f_merged = os.path.join(result_dir, "coco_pred_instances_merged.json")
    f_merged_exc = os.path.join(result_dir, "coco_pred_instances_excepted.json")

    # =========================================================================
    # [수정] 빈 JSON 파일(예측 없음) 처리용 헬퍼 함수
    # =========================================================================
    def safe_coco_eval_t1(gt, pred):
        """Table 1용 안전한 COCO 평가: 파일이 비어있으면 0점 반환"""
        try:
            with open(pred, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not data:  # 리스트가 비어있으면
                # AP 점수 0.0 딕셔너리와 인스턴스 0 반환
                return {k: 0.0 for k in ["0.10", "0.20", "0.50"]}, 0
            return t1.coco_ap_set_evalstyle(gt, pred)
        except Exception:
            # 혹시 모를 로드 에러 대비
            return {k: 0.0 for k in ["0.10", "0.20", "0.50"]}, 0

    def safe_coco_eval_t2(gt, pred):
        """Table 2용 안전한 COCO 평가: 파일이 비어있으면 빈 dict 반환"""
        try:
            with open(pred, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not data:
                return {}  # 빈 결과
            return t2.ap20_per_class_evalstyle(gt, pred)
        except Exception:
            return {}

    # =========================================================================

    # Table 1 생성
    seg_pixel_acc = t1.pixel_accuracy_from_dirs(label_dir, pred_dir_static)

    # 2. Linestrings Metrics (수정된 함수 사용)
    ap_all_lines, count_all = safe_coco_eval_t1(gt_json, f_origin)
    ap_long_lines, _ = safe_coco_eval_t1(gt_json, f_origin_exc)
    pix_acc_lines = t1.pixel_accuracy_from_json_vs_labels(f_origin, label_dir)

    # 3. Merged Metrics (수정된 함수 사용)
    ap_all_merged, count_merged = safe_coco_eval_t1(gt_json, f_merged)
    ap_long_merged, _ = safe_coco_eval_t1(gt_json, f_merged_exc)
    pix_acc_merged = t1.pixel_accuracy_from_json_vs_labels(f_merged, label_dir)

    t1_headers = ["", "instances", "AP10 (all)", "AP20 (all)", "AP50 (all)",
                  "AP10 (long)", "AP20 (long)", "AP50 (long)", "pixel accuracy"]
    f_p = lambda x: f"{x:.6f}" if (x is not None and np.isfinite(x)) else "-"

    t1_rows = [
        ["segmentation map", "-", "-", "-", "-", "-", "-", "-", f_p(seg_pixel_acc)],
        ["linestrings", str(count_all),
         f_p(ap_all_lines.get("0.10")), f_p(ap_all_lines.get("0.20")), f_p(ap_all_lines.get("0.50")),
         f_p(ap_long_lines.get("0.10")), f_p(ap_long_lines.get("0.20")), f_p(ap_long_lines.get("0.50")),
         f_p(pix_acc_lines)],
        ["linestrings merged", str(count_merged),
         f_p(ap_all_merged.get("0.10")), f_p(ap_all_merged.get("0.20")), f_p(ap_all_merged.get("0.50")),
         f_p(ap_long_merged.get("0.10")), f_p(ap_long_merged.get("0.20")), f_p(ap_long_merged.get("0.50")),
         f_p(pix_acc_merged)],
    ]

    with open(os.path.join(exp_path, "table_1.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(t1_headers)
        writer.writerows(t1_rows)

    # Table 2 생성
    target_json = f_merged

    # (수정된 함수 사용)
    ap20_by_cat = safe_coco_eval_t2(gt_json, target_json)
    pixacc_by_cat = t2.pixel_accuracy_per_class_from_json(target_json, label_dir)
    pred_counts = t2.count_pred_instances_by_cat(target_json)
    gt_counts = t2.count_gt_instances_by_cat(gt_json)

    t2_headers = ["class", "AP20 (long)", "pixel accuracy", "# pred instances", "# GT instances"]
    t2_rows = []

    for cid in t2.EVAL_CLASS_IDS:
        name = t2.ID2NAME.get(cid, str(cid))
        ap20 = ap20_by_cat.get(cid, float('nan'))
        pix = pixacc_by_cat.get(cid, float('nan'))
        pc = pred_counts.get(cid, 0)
        gc = gt_counts.get(cid, 0)

        t2_rows.append([
            name,
            f"{ap20:.6f}" if np.isfinite(ap20) else "-",
            f"{pix:.6f}" if np.isfinite(pix) else "-",
            str(pc),
            str(gc)
        ])

    with open(os.path.join(exp_path, "table_2.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(t2_headers)
        writer.writerows(t2_rows)


if __name__ == '__main__':
    run_experiments()