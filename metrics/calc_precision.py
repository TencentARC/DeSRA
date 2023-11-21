import argparse
import cv2
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_mask_root', type=str, help='the root of detected artifacts maps in GAN-SR results', required=True)
parser.add_argument(
    '--gt_mask_root', type=str, help='the root of human-labeled gt artifacts maps in GAN-SR results', required=True)
parser.add_argument('--threshold', type=float, default=0.5)
args = parser.parse_args()

total_detected_artifacts_nums = 0
real_artifacts_nums = 0

img_names = sorted(os.listdir(args.input_mask_root))
for img_name in img_names:
    input_mask_path = os.path.join(args.input_mask_root, img_name)
    input_mask = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)
    # detected mask
    detected_mask_num_labels, detected_mask_labels, detected_mask_stats, detected_mask_centroids = cv2.connectedComponentsWithStats(
        input_mask, connectivity=8)
    total_detected_artifacts_nums += (detected_mask_num_labels - 1)
    # gt mask
    gt_path = os.path.join(args.gt_mask_root, img_name)
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    h, w = gt_mask.shape
    gt_mask = gt_mask[5:h - 5, 5:w - 5]

    # loop
    image_filtered = np.zeros_like(input_mask)
    for (i, label) in enumerate(np.unique(detected_mask_labels)):
        if label == 0:
            continue
        image_filtered = np.zeros_like(input_mask)
        image_filtered[detected_mask_labels == i] = 1
        overlap_mask = np.asarray(image_filtered, dtype="int32") & np.asarray(gt_mask / 255, dtype="int32")
        ratio = np.sum(overlap_mask) / np.sum(image_filtered)
        if ratio >= args.threshold:
            real_artifacts_nums += 1

print(real_artifacts_nums / total_detected_artifacts_nums)
