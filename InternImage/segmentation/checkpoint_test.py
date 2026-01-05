import torch

# 체크포인트 로드
checkpoint_path = '/home/humpback/youn_ws/Internimage_rilab/InternImage/segmentation/work_dirs/upernet_internimage_l_512_160k_satellite/best_mIoU_iter_160000.pth'
ckpt = torch.load(checkpoint_path, map_location='cpu')

# 메타 정보 확인
meta = ckpt.get('meta', {})
print("=== Checkpoint Meta Info ===")
print("Epoch:", meta.get('epoch'))
print("Iteration:", meta.get('iter'))
print("mIoU:", meta.get('mIoU'))
print("CLASSES:", meta.get('CLASSES'))
print("PALETTE:", meta.get('PALETTE'))

# 모델 파라미터 키 개수
print("\nNumber of state_dict params:", len(ckpt['state_dict']))

# 첫 5개 파라미터 이름 출력
print("\nFirst 5 keys in state_dict:")
for i, key in enumerate(ckpt['state_dict'].keys()):
    if i >= 5:
        break
    print(f"  {key}")
