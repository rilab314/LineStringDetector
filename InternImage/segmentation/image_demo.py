# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from argparse import ArgumentParser

import cv2
import mmcv
import numpy as np
import mmcv_custom  # noqa: F401,F403
import mmseg_custom  # noqa: F401,F403
from mmcv.runner import load_checkpoint
from mmseg.apis import inference_segmentor, init_segmentor


def satellite_classes():
    return [
        'ignore', 'center_line', 'u_turn_zone_line', 'lane_line',
        'bus_only_lane', 'edge_line', 'path_change_restriction_line',
        'no_parking_stopping_line', 'guiding_line', 'stop_line',
        'safety_zone', 'bicycle_lane'
    ]


def satellite_palette():
    return [
        [0, 0, 0],            # ignore
        [77, 77, 255],        # center_line
        [77, 178, 255],       # u_turn_zone_line
        [77, 255, 77],        # lane_line
        [255, 153, 77],       # bus_only_lane
        [255, 77, 77],        # edge_line
        [178, 77, 255],       # path_change_restriction_line
        [77, 255, 178],       # no_parking_stopping_line
        [255, 178, 77],       # guiding_line
        [77, 102, 255],       # stop_line
        [255, 77, 128],       # safety_zone
        [128, 255, 77]        # bicycle_lane
    ]


def get_classes(dataset):
    if dataset == 'satellite':
        return satellite_classes()
    else:
        from mmseg.core import get_classes as mmseg_get_classes
        return mmseg_get_classes(dataset)


def get_palette(dataset):
    if dataset == 'satellite':
        return satellite_palette()
    else:
        from mmseg.core.evaluation import get_palette as mmseg_get_palette
        return mmseg_get_palette(dataset)


def apply_palette_to_mask(seg_mask, palette):
    """클래스 인덱스 마스크에 팔레트를 적용하여 RGB 이미지 생성"""
    color_mask = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in enumerate(palette):
        bgr_color = color[::-1]  # RGB → BGR
        color_mask[seg_mask == class_idx] = bgr_color
    return color_mask


def test_single_image(model, img_name, out_dir, color_palette):
    assumed_imgformat = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
    if not img_name.lower().endswith(assumed_imgformat):
        print(f'Skip {img_name} because it is not an image file.')
        return

    result = inference_segmentor(model, img_name)

    if hasattr(model, 'module'):
        model = model.module

    seg_mask = result[0].astype('uint8')
    seg_mask = np.clip(seg_mask - 1, 0, 255)
    color_mask = apply_palette_to_mask(seg_mask, color_palette)
    print(f"Seg mask unique values: {np.unique(seg_mask)}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, osp.splitext(osp.basename(img_name))[0] + '.png')
    cv2.imwrite(out_path, color_mask)
    print(f'Colored mask is saved at {out_path}')


def main():
    parser = ArgumentParser()
    parser.add_argument(
        'img', help='Image file or a directory contains images')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out', type=str, default='demo', help='out dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='ade20k',
        choices=['ade20k', 'cityscapes', 'cocostuff', 'satellite'],
        help='Color palette used for segmentation map')
    args = parser.parse_args()

    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)

    # print("Loaded classes:", model.CLASSES)

    if osp.isdir(args.img):
        for img_file in sorted(os.listdir(args.img)):
            img_path = osp.join(args.img, img_file)
            test_single_image(model, img_path, args.out, get_palette(args.palette))
    else:
        test_single_image(model, args.img, args.out, get_palette(args.palette))


if __name__ == '__main__':
    main()
