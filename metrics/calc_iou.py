import argparse
import cv2
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_mask_root', type=str, help='the root of detected artifacts maps in GAN-SR results', required=True)
parser.add_argument(
    '--gt_mask_root', type=str, help='the root of human-labeled gt artifacts maps in GAN-SR results', required=True)
args = parser.parse_args()

avg_iou = 0.0
count = 0

img_names = sorted(os.listdir(args.input_mask_root))
for img_name in img_names:
    input_mask_path = os.path.join(args.input_mask_root, img_name)
    input_mask = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE) / 255
    gt_path = os.path.join(args.gt_mask_root, img_name)
    if not os.path.exists(gt_path):
        print(input_mask_path)
        os.remove(f'{args.input_mask_root}/{img_name}')
        continue
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) / 255
    h, w = gt_mask.shape
    gt_mask = gt_mask[5:h - 5, 5:w - 5]
    overlap_mask = np.asarray(input_mask, dtype="int32") & np.asarray(gt_mask, dtype="int32")
    total_mask = np.asarray(input_mask, dtype="int32") | np.asarray(gt_mask, dtype="int32")
    iou = np.sum(overlap_mask) / np.sum(total_mask)

    count += 1
    avg_iou += iou

avg_iou /= count
print(avg_iou)
