import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_COCO_instseg(gt_path: str, pred_path: str) -> dict:
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(pred_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    metric_names = [
        'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',  # AP (Average Precision)
        'AR@1', 'AR@10', 'AR@100', 'ARs', 'ARm', 'ARl'  # AR (Average Recall)
    ]
    eval_results = {name: float(value) for name, value in zip(metric_names, stats)}

    return eval_results


# --- 아래는 함수 사용을 위한 예제 코드입니다 ---
if __name__ == '__main__':
    gt_path = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2025_LaneDetector/new_coco_dataset/annotations/instances_validation2017.json'
    pred_path = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2025_LaneDetector/ade20k/satellite_ade20k_250820/process/coco_pred_instances_merged.json'
    results = evaluate_COCO_instseg(gt_path, pred_path)

    print("-" * 30)
    print("평가 결과 (딕셔너리):")
    # 예쁘게 출력
    for key, value in results.items():
        print(f"{key:<7}: {value:.3f}")
