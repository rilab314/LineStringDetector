import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_COCO_instseg(gt_path: str, pred_path: str) -> dict:
    """
    COCO 형식의 Ground Truth와 예측 데이터를 사용하여 Instance Segmentation 성능을 평가합니다.

    Args:
        gt_path (str): COCO 형식의 Ground Truth JSON 파일 경로.
        pred_path (str): COCO 형식의 모델 예측 결과 JSON 파일 경로.

    Returns:
        dict: 평가 지표 이름과 값을 매핑한 딕셔너리.
              예: {'AP': 0.505, 'AP50': 0.895, ...}
    """
    # 1. COCO 객체 생성 및 데이터 로드
    # Ground Truth 용 COCO 객체 생성
    coco_gt = COCO(gt_path)

    # 예측 결과를 Ground Truth 객체에 로드
    # loadRes는 파일 경로 또는 파이썬 리스트를 인자로 받을 수 있습니다.
    coco_dt = coco_gt.loadRes(pred_path)

    # 2. COCOeval 객체 생성
    # iouType을 'segm'으로 설정하여 인스턴스 분할을 평가합니다.
    # (바운딩 박스는 'bbox')
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')

    # 3. 평가 실행
    coco_eval.evaluate()  # 이미지/카테고리별 평가 수행
    coco_eval.accumulate()  # 평가 결과 종합
    coco_eval.summarize()  # 터미널에 평가 결과 요약 출력

    # 4. 결과 추출 및 반환
    # coco_eval.stats는 12개의 평가 지표를 numpy 배열로 가지고 있습니다.
    stats = coco_eval.stats

    # 평가 지표 이름을 순서대로 정의
    metric_names = [
        'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',  # AP (Average Precision)
        'AR@1', 'AR@10', 'AR@100', 'ARs', 'ARm', 'ARl'  # AR (Average Recall)
    ]

    # 지표 이름과 값을 짝지어 딕셔너리로 만듭니다.
    eval_results = {name: float(value) for name, value in zip(metric_names, stats)}

    return eval_results


# --- 아래는 함수 사용을 위한 예제 코드입니다 ---
if __name__ == '__main__':
    # 이 예제를 실행하려면 pycocotools가 필요합니다.
    # Linux/MacOS: pip install pycocotools
    # Windows: pip install pycocotools-windows

    # 1. 가상의 Ground Truth 데이터 생성 (실제로는 파일이 이미 존재)
    gt_data = {
        "images": [
            {"id": 1, "width": 640, "height": 480, "file_name": "test_image.jpg"}
        ],
        "annotations": [
            {
                "id": 101,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[100, 100, 200, 100, 200, 200, 100, 200]],  # 사각형 폴리곤
                "area": 10000,
                "bbox": [100, 100, 100, 100],
                "iscrowd": 0
            }
        ],
        "categories": [
            {"id": 1, "name": "cat", "supercategory": "animal"}
        ]
    }
    with open('ground_truth.json', 'w') as f:
        json.dump(gt_data, f)

    # 2. 가상의 모델 예측 데이터 생성 (실제로는 모델이 출력한 파일)
    # 예측 결과는 score 필드를 반드시 포함해야 합니다.
    pred_data = [
        {
            "image_id": 1,
            "category_id": 1,
            "segmentation": [[105, 105, 195, 105, 195, 195, 105, 195]],  # GT와 거의 겹침
            "score": 0.95  # 높은 신뢰도 점수
        }
    ]
    with open('predictions.json', 'w') as f:
        json.dump(pred_data, f)

    print("가상 데이터로 평가를 시작합니다...")
    print("-" * 30)

    # 3. 함수 호출 및 결과 확인
    results = evaluate_COCO_instseg('ground_truth.json', 'predictions.json')

    print("-" * 30)
    print("평가 결과 (딕셔너리):")
    # 예쁘게 출력
    for key, value in results.items():
        print(f"{key:<7}: {value:.3f}")
