import cv2
import numpy as np


def read_image(im_path: str) -> np.ndarray:
    image = cv2.imread(im_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image
