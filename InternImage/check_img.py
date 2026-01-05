import cv2
import numpy as np
import os


def get_unique_colors_from_folder(folder_path):
    unique_colors = set()
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))]

    total_images = len(image_files)
    for idx, image_file in enumerate(image_files, start=1):
        print(f"Processing image {idx}/{total_images}: {image_file}")
        img_path = os.path.join(folder_path, image_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            reshaped_img = img.reshape(-1)
            for color in reshaped_img:
                unique_colors.add(color)

        print(f"Unique colors so far: {unique_colors}")

    return unique_colors


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        gray_color = param[y, x]
        print(f"Coordinates: ({x}, {y}), Grayscale Color: {gray_color}")


def main():
    image_folder = '/home/humpback/youn_ws/Internimage_rilab/data/ADE20K_2021_16/ADEChallengeData2016/annotations/training'  # 여기에 이미지 포르더 경로를 입력하세요

    # 포르더의 모든 이미지에서 고유 색상 추출 후 출력
    all_unique_colors = get_unique_colors_from_folder(image_folder)
    print(f"All unique colors in folder: {all_unique_colors}")

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))]
    image_files.sort()

    total_images = len(image_files)
    for idx, image_file in enumerate(image_files, start=1):
        print(f"Displaying image {idx}/{total_images}: {image_file}")
        img_path = os.path.join(image_folder, image_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # 현재 이미지의 고유 색상 출력
        unique_colors = np.unique(img)
        print(f"Unique colors in {image_file}: {unique_colors}")

        cv2.imshow('Image', img)
        cv2.setMouseCallback('Image', click_event, param=img)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('n'):  # 'n' 키를 누르면 다음 이미지로
                break
            elif key == ord('q'):  # 'q' 키를 누르면 종료
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
