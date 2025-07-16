#!/usr/bin/env python3
import os
from collections import defaultdict

# ====== 설정 부분 ======
# 이미지가 저장된 폴더 경로를 문자열로 입력하세요.
folder_path = "/home/humpback/youn_ws/Internimage_rilab/data/satellite_good_matching_241125/image"

# 소수점 반올림 자릿수를 정수로 입력하세요.
decimals = 3

# ====== 함수 정의 ======
def find_duplicate_coords(folder_path, decimals=3):
    """
    주어진 폴더에서 '경도,위도.png' 형식의 파일명을 가진 PNG 파일을 스캔한 후,
    좌표를 소수점 N자리로 반올림하여 중복되는 파일명을 출력합니다.
    """
    coord_map = defaultdict(list)

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith('.png'):
            continue
        name = filename[:-4]  # '.png' 확장자 제거
        try:
            lon_str, lat_str = name.split(',')
            lon = float(lon_str)
            lat = float(lat_str)
        except ValueError:
            # 파일명이 "경도,위도" 패턴이 아닐 경우 건너뜀
            continue

        # 좌표를 소수점 N자리로 반올림
        lon_r = round(lon, decimals)
        lat_r = round(lat, decimals)

        coord_map[(lon_r, lat_r)].append(filename)

    # 중복된 좌표를 가진 파일 그룹 출력
    duplicates = {coords: files for coords, files in coord_map.items() if len(files) > 1}
    if duplicates:
        print(f"중복된 좌표 (소수점 {decimals}자리 반올림 기준):")
        for (lon_r, lat_r), files in duplicates.items():
            # 출력 시에도 고정 소수점 자릿수로 표시
            print(f"  ({lon_r:.{decimals}f}, {lat_r:.{decimals}f}) -> {files}")
    else:
        print(f"소수점 {decimals}자리로 반올림했을 때 중복된 좌표가 없습니다.")

# ====== 실행 ======
if __name__ == "__main__":
    find_duplicate_coords(folder_path, decimals)

