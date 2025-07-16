import os
import cv2
import glob
import shutil

# 경로 설정
origin_img_path = "/home/humpback/youn_ws/LaneDetector_rilab/dataset/mask2former/images"  # 원본 이미지 폴더
img_save_path = "/home/humpback/youn_ws/LaneDetector_rilab/dataset/mask2former/extract_images/images"  # 원본 이미지 저장 폴더
pred_img_path = "/home/humpback/youn_ws/LaneDetector_rilab/dataset/mask2former/pred_images"  # 마스크 이미지 폴더
pred_save_path = "/home/humpback/youn_ws/LaneDetector_rilab/dataset/mask2former/extract_images/pred_images"  # 마스크 저장 폴더
npy_path = "/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2025_LaneDetector/mask2former/segmap"  # .npy 파일이 있는 폴더
save_npy_path = "/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2025_LaneDetector/mask2former/extract_segmap"  # .npy 파일 저장 폴더

# 폴더가 없으면 생성
os.makedirs(img_save_path, exist_ok=True)
os.makedirs(pred_save_path, exist_ok=True)
os.makedirs(save_npy_path, exist_ok=True)

# 원본 이미지 목록 가져오기
image_files = glob.glob(os.path.join(origin_img_path, "*"))

for img_file in sorted(image_files):  # 파일 정렬 후 반복
    img = cv2.imread(img_file)  # 원본 이미지 로드
    if img is None:
        print(f"이미지를 불러올 수 없음: {img_file}")
        continue

    # 파일명만 추출
    filename = os.path.basename(img_file)
    pred_file = os.path.join(pred_img_path, filename)  # pred_img 폴더에서 같은 이름의 마스크 찾기
    npy_file = os.path.join(npy_path, filename.replace(os.path.splitext(filename)[1], ".npy"))  # 같은 이름의 .npy 찾기

    cv2.imshow("Image Viewer", img)  # 원본 이미지 표시
    key = cv2.waitKey(0)  # 키 입력 대기

    if key == ord('s'):  # 's' 키를 누르면 저장
        save_img_path = os.path.join(img_save_path, filename)
        cv2.imwrite(save_img_path, img)
        print(f"이미지 저장됨: {save_img_path}")

        # 마스크 파일이 존재하면 pred_save_path 폴더에 저장
        if os.path.exists(pred_file):
            save_pred_path = os.path.join(pred_save_path, filename)
            shutil.copy(pred_file, save_pred_path)
            print(f"마스크 저장됨: {save_pred_path}")
        else:
            print(f"마스크 없음: {pred_file}")

        # 같은 이름의 .npy 파일이 존재하면 save_npy_path 폴더에 저장
        if os.path.exists(npy_file):
            save_npy_file = os.path.join(save_npy_path, os.path.basename(npy_file))
            shutil.copy(npy_file, save_npy_file)
            print(f"npy 저장됨: {save_npy_file}")
        else:
            print(f"npy 파일 없음: {npy_file}")

    elif key == 27:  # ESC 키를 누르면 종료
        break

cv2.destroyAllWindows()  # 모든 창 닫기
