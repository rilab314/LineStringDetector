import cv2
import os
import numpy as np


class ImageShow:
    pad = 10
    title_pad = 25

    def __init__(self, window_title: str, columns: int, scale: float = 1.0, enabeled=False):
        self.save_path = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2025_LaneDetector/class_img'
        self._window_title = window_title
        self._columns = columns
        self._scale = scale
        self._titled_imgs = {}
        self._enabeled = enabeled

    def show(self, image: np.ndarray, title: str, wait_ms: int = 0, dilate: bool = False, file_name: str = None, class_id: int = None):
        if not self._enabeled:
            if dilate:
                image = cv2.dilate(image, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8), iterations=1)
            label_image = image.copy()
            self._titled_imgs[title] = image
            image = self.update_whole_image()
            image = self.scale_image(image)
            if file_name is not None and class_id is None:
                file_only = os.path.basename(file_name)
                file_png = os.path.splitext(file_only)[0] + ".png"
                folder_path = os.path.join(self.save_path, 'all_class_merged')
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                cv2.imwrite(os.path.join(folder_path, file_png), label_image)

            if file_name is not None and class_id is not None:
                file_only = os.path.basename(file_name)
                file_png = os.path.splitext(file_only)[0] + ".png"
                folder_path = os.path.join(self.save_path, str(class_id))
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                cv2.imwrite(os.path.join(folder_path, file_png), image)

            cv2.imshow(self._window_title, image)
            cv2.waitKey(wait_ms)
    
    def remove(self, title, wait_ms: int = 0):
        if not self._enabeled:
            if isinstance(title, str):
                if title in self._titled_imgs:
                    del self._titled_imgs[title]
                else:
                    # print(f'[ImageShow.remove] {title}
                    pass
            elif isinstance(title, list):
                for name in title:
                    if name in self._titled_imgs:
                        del self._titled_imgs[name]
                    else:
                        # print(f'[ImageShow.remove] {name} is not in titled_imgs')
                        continue
            image = self.update_whole_image()
            image = self.scale_image(image)
            cv2.imshow(self._window_title, image)
            cv2.waitKey(wait_ms)

    def update_whole_image(self):
        titled_imgs = self.gray_to_bgr(self._titled_imgs)
        block_size, image_size = self.get_block_info(titled_imgs)
        result_img = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8)*255
        for idx, (title, image) in enumerate(titled_imgs.items()):
            y_pos = int(idx / self._columns) * block_size[0] + self.title_pad
            x_pos = int(idx % self._columns) * block_size[1] + self.pad
            img_shape = image.shape
            result_img[y_pos:y_pos+img_shape[0], x_pos:x_pos+img_shape[1], :] = image
            title_pos = (x_pos - self.pad, y_pos - 5)
            cv2.putText(result_img, title, title_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        return result_img

    def gray_to_bgr(self, images):
        bgr_imgs = dict()
        for key, img in images.items():
            if len(img.shape) == 2:
                bgr_imgs[key] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                bgr_imgs[key] = img.copy()
        return bgr_imgs

    def get_block_info(self, imgs_dict):
        max_height, max_width = 0, 0
        for img in imgs_dict.values():
            height, width = img.shape[:2]
            max_height = max(max_height, height)
            max_width = max(max_width, width)
        
        block_size = (max_height + self.title_pad + self.pad, max_width + self.pad*2)
        rows = (len(imgs_dict) - 1) // self._columns + 1
        image_size = (block_size[0]*rows, block_size[1]*self._columns)
        return block_size, image_size
    
    def scale_image(self, image: np.ndarray):
        return cv2.resize(image, None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_NEAREST)
