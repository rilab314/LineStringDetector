import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import config as cfg


def _ap_at_iou_from_eval(coco_eval: COCOeval, iou: float) -> float:
    """
    이미 evaluate/accumulate가 끝난 coco_eval 객체에서
    특정 IoU의 AP를 precision 배열에서 직접 계산해 반환.
    """
    ths = coco_eval.params.iouThrs  # shape: [T]
    idx = np.where(np.isclose(ths, iou))[0]
    if idx.size == 0:
        return float('nan')
    # precision shape: [T, R, K, A, M]
    p = coco_eval.eval['precision'][idx[0]]  # [R, K, A, M]
    p = p[p > -1]
    return float(np.mean(p)) if p.size else float('nan')


def evaluate_COCO_instseg(gt_path: str, pred_path: str) -> dict:
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(pred_path)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')

    # 방법 B: 여러 IoU를 한 번에 설정 (0.10 ~ 0.20까지 0.01 간격, 포함)
    coco_eval.params.iouThrs = np.linspace(0.10, 0.20, 11, dtype=np.float32)

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()  # 표준 12개 통계를 출력(참고용). AP50/AP75는 -1로 나올 수 있음

    # 표준 stats는 COCO 고정 의미: [AP@[.5:.95], AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]
    stats = coco_eval.stats  # 그대로 참고용으로도 반환
    # 방법 B 핵심: 원하는 IoU에서의 AP를 precision 배열에서 직접 산출
    ap10 = _ap_at_iou_from_eval(coco_eval, 0.10)
    ap20 = _ap_at_iou_from_eval(coco_eval, 0.20)

    # IoU=0.10~0.20 범위 평균 AP(참고용)도 함께 계산 가능
    # (precision을 직접 훑어 평균내도 되고, 여기서는 각 임계값별 AP를 평균)
    ths = coco_eval.params.iouThrs
    aps = []
    for th in ths:
        aps.append(_ap_at_iou_from_eval(coco_eval, float(th)))
    ap_range_mean = float(np.nanmean(aps)) if len(aps) else float('nan')

    eval_results = {
        # 방법 B로 구한 값
        'AP10': ap10,
        'AP20': ap20,
        'AP@.10:.20(mean)': ap_range_mean,
        # 참고: COCO 기본 stats (의미는 고정, 라벨 바꾸지 말 것)
        'COCO_AP@[.50:.95]': float(stats[0]),
        'COCO_AP50': float(stats[1]),
        'COCO_AP75': float(stats[2]),
        'COCO_APs': float(stats[3]),
        'COCO_APm': float(stats[4]),
        'COCO_APl': float(stats[5]),
        'COCO_AR@1': float(stats[6]),
        'COCO_AR@10': float(stats[7]),
        'COCO_AR@100': float(stats[8]),
        'COCO_ARs': float(stats[9]),
        'COCO_ARm': float(stats[10]),
        'COCO_ARl': float(stats[11]),
    }
    return eval_results


# --- 아래는 함수 사용을 위한 예제 코드입니다 ---
if __name__ == '__main__':
    gt_path = cfg.COCO_ANNO_PATH
    pred_path = cfg.MERGED_JSON_PATH

    results = evaluate_COCO_instseg(gt_path, pred_path)

    print("-" * 30)
    print("평가 결과 (딕셔너리):")
    # 보기 좋게 정렬 출력
    order = [
        'AP10', 'AP20', 'AP@.10:.20(mean)',
        'COCO_AP@[.50:.95]', 'COCO_AP50', 'COCO_AP75',
        'COCO_APs', 'COCO_APm', 'COCO_APl',
        'COCO_AR@1', 'COCO_AR@10', 'COCO_AR@100',
        'COCO_ARs', 'COCO_ARm', 'COCO_ARl'
    ]
    for k in order:
        v = results.get(k, float('nan'))
        print(f"{k:<18}: {v:.3f}" if isinstance(v, float) else f"{k:<18}: {v}")
