import hashlib

import numpy as np

from obj_det import io


def test_read_image():
    image = io.read_image("tests/files/test_im.jpg")

    assert image is not None
    assert type(image) == np.ndarray

    image_str = np.array2string(image)
    assert len(image_str) == 629
    image_hash = hashlib.sha1(image_str.encode()).hexdigest()

    assert image_hash == "cb0b5c22bff5c02f95730defc73c6a35c0561561"
