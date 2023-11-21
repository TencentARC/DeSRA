import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from scipy import ndimage

from distance import calc_artifact_map
from mmseg.apis import inference_segmentor, init_segmentor, show_seg
from mmseg.core.evaluation import get_palette

CLASSES = [
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'windowpane', 'grass', 'cabinet', 'sidewalk',
    'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa',
    'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp',
    'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand',
    'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table',
    'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower',
    'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
    'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight',
    'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator',
    'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer',
    'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven',
    'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher',
    'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier',
    'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag'
]


# The version of mmsegmentation is '0.29.0'
# The version of mmcv is '1.6.1'
def main():
    parser = ArgumentParser()
    parser.add_argument('--mse_root', required=True, help='Root of MSE images')
    parser.add_argument('--gan_root', required=True, help='Root of GAN images')
    parser.add_argument('--save_root', required=True, help='Path to output file')
    # Divide the value interval from 0 to 1 into 20 sub-intervals. For each pixel under a semantic class,
    # calculate in which of the first few intervals the D value of more than 85% of the pixels is distributed.
    # You can refer to Figure 10(b) in our paper.
    parser.add_argument('--semantic_interval_txt', default='./SSIM_MSE_GAN_threshold.txt')
    parser.add_argument('--contrast_threshold', help='threshold of contrast', default=0.7, type=float)
    parser.add_argument('--area_threshold', help='threshold of minimal area', default=4000, type=int)
    parser.add_argument(
        '--config', help='Config file', default='configs/segformer/segformer_mit-b5_640x640_160k_ade20k.py')
    parser.add_argument(
        '--checkpoint',
        help='Checkpoint file',
        default='checkpoints/segformer_mit-b5_640x640_160k_ade20k_20220617_203542-940a6bd8.pth')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--palette', default='ade', help='Color palette used for segmentation map')
    parser.add_argument('--window_size', default=11, type=int)
    args = parser.parse_args()

    # configuration
    window_size = args.window_size
    contrast_threshold = args.contrast_threshold
    area_threshold = args.area_threshold
    os.makedirs(args.save_root, exist_ok=True)

    # calculate the adjustment weight for each semantic class. Refer to Figure 10(b) in our paper.
    adjustment_weight_dict = {}
    semantic_interval_content = open(args.semantic_interval_txt, 'r').readlines()
    for line in semantic_interval_content:
        line = line.strip()
        basename, idx = line.split('--')
        adjustment_weight_dict[basename] = 20 / (20 - int(idx))

    # initialize the segformer
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    for sub_root_name in sorted(os.listdir(args.mse_root)):
        sub_path = os.path.join(args.mse_root, sub_root_name)
        for img_name in sorted(os.listdir(sub_path)):
            print(sub_root_name, img_name)
            mse_img_path = os.path.join(args.mse_root, sub_root_name, img_name)
            gan_img_path = os.path.join(args.gan_root, sub_root_name, img_name)

            print('We are processing: ', f'{sub_root_name}/{img_name}')

            if not os.path.exists(gan_img_path):
                continue

            save_final_mask_root = os.path.join(args.save_root,
                                                f'contrast_{contrast_threshold}_threshold_{area_threshold}',
                                                sub_root_name)
            os.makedirs(save_final_mask_root, exist_ok=True)
            save_final_mask_path = os.path.join(save_final_mask_root, img_name)
            if os.path.exists(save_final_mask_path):
                continue
            # === First step: generate the segmentation map for GAN-SR image. ===
            save_seg_mask_root = os.path.join(args.save_root, 'seg', sub_root_name)
            os.makedirs(save_seg_mask_root, exist_ok=True)
            save_seg_mask_path = os.path.join(save_seg_mask_root, img_name)

            try:
                with torch.no_grad():
                    result = inference_segmentor(model, gan_img_path)
            except Exception as error:
                print('Error', error, img_name)
            else:
                seg_mask = show_seg(model, gan_img_path, result, get_palette(args.palette))

                # save mask
                cv2.imwrite(save_seg_mask_path, seg_mask)

                # === Second step: Relative difference of local variance between MSE-SR and GAN-SR patches. ===
                # Refer to Equation 7 in our paper.
                save_contrast_root = os.path.join(args.save_root, 'rgb_contrast', sub_root_name)
                os.makedirs(save_contrast_root, exist_ok=True)
                save_contrast_path = os.path.join(save_contrast_root, img_name)

                mse_img = cv2.imread(mse_img_path, cv2.IMREAD_UNCHANGED)
                gan_img = cv2.imread(gan_img_path, cv2.IMREAD_UNCHANGED)
                artifact_map = calc_artifact_map(mse_img, gan_img, crop_border=0, window_size=window_size)

                cv2.imwrite(save_contrast_path, artifact_map * 255)

                # === Third step: Semantic-aware adjustment. ===
                # Refer to Equation 9 in our paper.
                save_contrast_seg_root = os.path.join(args.save_root, 'rgb_contrast_seg', sub_root_name)
                os.makedirs(save_contrast_seg_root, exist_ok=True)
                save_contrast_seg_path = os.path.join(save_contrast_seg_root, img_name)

                labels = set(list(seg_mask.flatten()))
                h, w = seg_mask.shape
                seg_mask = seg_mask[window_size // 2:h - window_size // 2, window_size // 2:w - window_size // 2]
                semantic_threshold_mask = np.ones_like(seg_mask).astype(np.float64)
                for label in labels:
                    label_name = CLASSES[label]
                    semantic_threshold_mask[seg_mask == label] = adjustment_weight_dict[label_name]

                contrast_seg = artifact_map * semantic_threshold_mask  # range [0, 1]
                contrast_seg = np.clip(contrast_seg, 0, 1)
                cv2.imwrite(save_contrast_seg_path, contrast_seg * 255)

                contrast_seg_mask = np.zeros(contrast_seg.shape)
                contrast_seg_mask[contrast_seg < contrast_threshold] = 1

                # === Four step: Morphological operations. ===
                save_dst_root = os.path.join(args.save_root,
                                             f'contrast_{contrast_threshold}_erode_5x5_dilation_5x5_hole_3x3',
                                             sub_root_name)
                os.makedirs(save_dst_root, exist_ok=True)
                save_dst_path = os.path.join(save_dst_root, img_name)
                kernel = np.ones((5, 5), np.uint8)
                erosion = cv2.erode(contrast_seg_mask, kernel, iterations=1)
                dilation = cv2.dilate(erosion, kernel, iterations=3)
                dst = ndimage.binary_fill_holes(dilation, structure=np.ones((3, 3))).astype(int)
                cv2.imwrite(save_dst_path, dst * 255)

                save_final_mask_root = os.path.join(args.save_root, 'Final_Artifact_Map', sub_root_name)
                os.makedirs(save_final_mask_root, exist_ok=True)
                save_final_mask_path = os.path.join(save_final_mask_root, img_name)
                dst = cv2.imread(save_dst_path, cv2.IMREAD_GRAYSCALE)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dst, connectivity=8)
                image_filtered = np.zeros_like(dst)
                for (i, label) in enumerate(np.unique(labels)):
                    if label == 0:
                        continue
                    if stats[i][-1] > area_threshold:
                        image_filtered[labels == i] = 255

                cv2.imwrite(save_final_mask_path, image_filtered)


if __name__ == '__main__':
    main()
