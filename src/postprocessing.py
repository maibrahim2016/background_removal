from typing import Optional, Union

import cv2
import numpy as np
from sklearn.cluster import KMeans


def postprocess(image: np.ndarray, threshold: Optional[Union[str, int]] = None) -> np.ndarray:
    """Apply postprocessing on alpha prediction to make it appear better.
    Currently, the postprocessing steps are thresholding to make border sharper/crisper and 
    Gaussian blurring to make the border edges smoother.

    Arguments:
        image {np.ndarray} -- Image

    Keyword Arguments:
        threshold {Union[str, int]} -- Threshold used: int or "kmeans" (default: {"kmeans"})

    Returns:
        np.ndarray -- Processed image
    """

    print("Applying postprocessing...")

    ori_image_shape = image.shape

    # print(
    #     f">>>>>>>>>> image: min: {image.min()}, max: {image.max()}, dtype: {image.dtype}, shape: {image.shape}")

    if threshold:
        if threshold == "kmeans":
            kmeans = KMeans(n_clusters=2, random_state=0,
                            n_jobs=-1).fit(image.reshape(-1, 1))
            threshold = kmeans.cluster_centers_.ravel().mean()
        threshold = 128

        # image[image < threshold] = 0
        # image[image >= threshold] = 255
        _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    image = cv2.GaussianBlur(image, (5, 5), 0)
    print(
        f">>>>>>>>>> image: min: {image.min()}, max: {image.max()}, dtype: {image.dtype}, shape: {image.shape}")

    # image = (image * 255).astype("uint8")
    # _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
    # image = image / np.float32(255)

    if len(ori_image_shape) == 3 and image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    assert image.shape == ori_image_shape, "Image shape changed!"

    return image
