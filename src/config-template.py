import os

BASE_PATH = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Archive/Dataset/unzips/LaneDetector/ade20k'

DATA_PATH = os.path.join(BASE_PATH)
LABEL_PATH = os.path.join(BASE_PATH, 'annotations', 'validation')
PRED_PATH = os.path.join(BASE_PATH, 'prediction')
COCO_ANNO_PATH = os.path.join(BASE_PATH.replace('ade20k', 'coco'), 'annotations', 'instances_validation2017.json')
ORIGIN_JSON_PATH = os.path.join(BASE_PATH, 'result', 'coco_pred_instances_origin.json')
ORIGIN_EXCEPTED_JSON_PATH = os.path.join(BASE_PATH, 'result', 'coco_pred_instances_origin_excepted.json')
MERGED_JSON_PATH = os.path.join(BASE_PATH, 'result', 'coco_pred_instances_merged.json')
MERGED_EXCEPTED_JSON_PATH = os.path.join(BASE_PATH, 'result', 'coco_pred_instances_excepted.json')
RESULT_PATH = os.path.join(BASE_PATH, 'result')