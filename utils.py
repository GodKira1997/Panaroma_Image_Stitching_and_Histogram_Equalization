import numpy as np


def uint8_to_float32(image: np.ndarray) -> np.ndarray:
    return (image / 255).astype(np.float32)


def float32_to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.round(image * 255), 0, 255).astype(np.uint8)
