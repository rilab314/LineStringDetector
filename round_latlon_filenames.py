import os
import glob

# 🔹 변경할 이미지들이 있는 폴더 경로 (수정 가능)
image_folder = "/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2024_SATELLITE/datasets/satellite_ade20k_250310/annotations/validation"  # 이미지가 있는 폴더 경로 입력

# 🔹 폴더 내 모든 PNG 파일 찾기
image_files = glob.glob(os.path.join(image_folder, "*.png"))

for file_path in image_files:
    file_name = os.path.basename(file_path)  # 파일명만 추출 (예: 126.721585,37.500701.png)

    try:
        # 🔹 위경도 값과 확장자 분리
        coord_part, ext = file_name.rsplit(".", 1)  # "126.721585,37.500701" / "png"
        lon, lat = coord_part.split(",")  # 경도, 위도 분리

        # 🔹 소수점 4자리까지 반올림
        new_lon = round(float(lon), 4)
        new_lat = round(float(lat), 4)

        # 🔹 새로운 파일명 생성
        new_file_name = f"{new_lon},{new_lat}.png"
        new_file_path = os.path.join(image_folder, new_file_name)

        # 🔹 파일 이름 변경
        os.rename(file_path, new_file_path)
        print(f"Renamed: {file_name} → {new_file_name}")

    except Exception as e:
        print(f"Error processing {file_name}: {e}")

print("✅ 모든 파일이 소수점 4자리까지 반올림되어 변경되었습니다.")
