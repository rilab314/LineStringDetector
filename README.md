# LanePostProcess

1. 원본 데이터셋 다운로드 ( goodmatching, pretrained_ckpt )
2. mask2former 폴더 내부 ade20k, coco dataset builder로 ade20k, coco 데이터셋 생성
3. 생성된 ade20k 데이터셋 기준으로 internimage, mask2former 학습
4. 학습 완료 후 생성된 work_dir에 생성된 checkpoint로 inference 진행
5. inference를 통해 생성된 demo 폴더의 segmentation 이미지를 prediction으로 활용
6. ade20k 데이터셋 + segementation으로 lanedetector 진행
7. lanedetector 완료 후 table 제작에 coco dataset 사용




hf upload-large-folder --repo-type=dataset goodgodgd/SatelliteLaneDataset2024 .

