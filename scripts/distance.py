import cv2
import numpy as np


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def calc_artifact_map(img, img2, crop_border, input_order='HWC', window_size=11, **kwargs):
    """Calculate quantitative indicator in Equation 7.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: artifact map between two images.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    artifact_maps = []
    for i in range(img.shape[2]):
        indicator = calc_single_artifact_map(img[..., i], img2[..., i], window_size)
        artifact_maps.append(indicator)

    artifact_maps = np.stack(artifact_maps, axis=0)
    # mean
    artifact_map = np.mean(artifact_maps, axis=0)

    return artifact_map


def calc_single_artifact_map(img, img2, window_size=11):
    """The proposed quantitative indicator in Equation 7.

    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: artifact map of a single channel.
    """

    constant = (0.03 * 255)**2
    kernel = cv2.getGaussianKernel(window_size, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[window_size // 2:-(window_size // 2),
                                        window_size // 2:-(window_size // 2)]  # valid mode for window size 11
    mu2 = cv2.filter2D(img2, -1, window)[window_size // 2:-(window_size // 2), window_size // 2:-(window_size // 2)]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[window_size // 2:-(window_size // 2),
                                                 window_size // 2:-(window_size // 2)] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[window_size // 2:-(window_size // 2),
                                                  window_size // 2:-(window_size // 2)] - mu2_sq

    contrast_map = (2 * (sigma1_sq + 1e-8)**0.5 * (sigma2_sq + 1e-8)**0.5 + constant) / (
        sigma1_sq + sigma2_sq + constant)

    return contrast_map
