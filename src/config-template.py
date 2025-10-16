import os
BASE_PATH = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2025_LaneDetector/ade20k/setting/satellite_ade20k_250925_internimage_large'
WORK_PATH = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2025_LaneDetector/ade20k/setting/satellite_ade20k_250925_internimage_large'

THICKNESS = 6

DATA_PATH = os.path.join(BASE_PATH)
ORIGIN_PATH = os.path.join(BASE_PATH, 'images', 'validation')
GT_PATH = os.path.join(BASE_PATH, 'color_annotations', 'validation')
LABEL_PATH = os.path.join(BASE_PATH, 'annotations', 'validation')
PRED_PATH = os.path.join(BASE_PATH, 'prediction')
RESULT_PATH = os.path.join(BASE_PATH, "results", f'thickness={THICKNESS}')
MERGED_JSON_PATH = os.path.join(RESULT_PATH, 'coco_pred_instances_merged.json')
MERGED_EXCEPTED_JSON_PATH = os.path.join(RESULT_PATH, 'coco_pred_instances_excepted.json')
ORIGIN_JSON_PATH = os.path.join(RESULT_PATH, 'coco_pred_instances_origin.json')
ORIGIN_EXCEPTED_JSON_PATH = os.path.join(RESULT_PATH, 'coco_pred_instances_origin_excepted.json')

COCO_PATH = f'/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2025_LaneDetector/coco/satellite_coco_250908/thickness={THICKNESS}'
COCO_ANNO_PATH = os.path.join(COCO_PATH, 'annotations', 'instances_validation2017.json')

