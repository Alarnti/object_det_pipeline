from typing import List, Tuple

import pandas as pd
import numpy as np

from io import read_image


def show_im_with_bb(
    im_path: str, bbox: Tuple[float, float, float, float]
) -> np.ndarray:
    image = read_image(im_path)

    bbox_int = np.array(bbox).astype(int)
    start_point = tuple(bbox_int[:2])
    end_point = (start_point[0] + bbox_int[2], start_point[1] + bbox_int[3])

    image_with_bb = cv2.rectangle(image, start_point, end_point, (255, 0, 0), 2)

    return image_with_bb


def shom_im_from_df_fruit_zindi(
    folder_path: str, im_label: str, df: pd.DataFrame
) -> np.ndarray:
    im_path = folder_path + im_label + ".jpg"

    bbox_gt = tuple(
        df[df["Image_ID"] == im_label]
        .loc[:, ["xmin", "ymin", "width", "height"]]
        .values[0]
    )

    return show_im_with_bb(im_path, bbox_gt)
