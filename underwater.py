import cv2 as cv
import numpy as np


_DEFAULT_GAIN_BGR = (1.0, 1.0, 1.0)
_DEFAULT_ALPHA_LUT = 1.0


def adjust_gain_bgr(image: np.ndarray, scale: tuple[float, float, float]) -> np.ndarray:
    """Scale the B, G, and R channels of an image by the given scale factors. Input image should be
    in uint8 format. Output image will also be in uint8 format. Values are rounded and clipped to
    ensure that they are in the range [0, 255].
    """
    return np.clip(np.round(image * scale), 0, 255).astype(np.uint8)


def create_lookup_table(image: np.ndarray, alpha: float) -> np.ndarray:
    """Using numpy, create a lookup table for partial histogram equalization. When alpha=0,
    the lookup table should act such that the image is unchanged. When alpha=1, the lookup table
    should act such that the image is fully equalized. Values of alpha in between should act as a
    partial equalization.

    Useful numpy functions: np.arange(), np.bincount(), np.cumsum()
    """
    identity = np.arange(256, dtype=np.uint8)
    histogram = np.bincount(image.ravel(), minlength=256)
    lut = np.cumsum(histogram)
    return np.round(alpha * ((lut / lut.max()) * 255) + (1 - alpha) * identity).astype(np.uint8)


def enhance(
    input_image_path: str,
    output_image_path: str,
    gain_bgr: tuple[float, float, float],
    alpha_lut: float,
):
    """Main function for underwater image enhancement.

    Steps:
    1. apply scale factors to the B, G, and R channels of the image. chosen scale factors should
       do something like boost the amount of red in the image, and reduce the amount of blue and
       green, since water absorbs red light more than blue and green.
    2. convert the image to the LAB color space.
    3. create a lookup table that will partially equalize the L channel using the
       create_lookup_table function. Choose an alpha value that you think makes the image look good.
    4. apply the lookup table to the L channel using the cv.LUT function.
    5. convert the result back to the BGR color space.
    """
    image = cv.imread(input_image_path)
    if image is None:
        raise FileNotFoundError(f"Could not find image {input_image_path}")

    adjusted_colored_image = adjust_gain_bgr(image, gain_bgr)
    lab_image = cv.cvtColor(adjusted_colored_image, cv.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv.split(lab_image)
    lookup_table = create_lookup_table(l_channel, alpha_lut)
    equalized_l_channel = cv.LUT(l_channel, lookup_table)
    equalized_image = np.clip(np.round(np.dstack((equalized_l_channel, a_channel, b_channel))),
                              0, 255).astype(np.uint8)
    output_image = cv.cvtColor(equalized_image, cv.COLOR_Lab2BGR)
    cv.imwrite(output_image_path, output_image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str, help="input image")
    parser.add_argument("--output", required=True, type=str, help="output image")
    parser.add_argument(
        "--gain-b",
        required=False,
        type=float,
        default=_DEFAULT_GAIN_BGR[0],
        help="gain for blue channel",
    )
    parser.add_argument(
        "--gain-g",
        required=False,
        type=float,
        default=_DEFAULT_GAIN_BGR[1],
        help="gain for green channel",
    )
    parser.add_argument(
        "--gain-r",
        required=False,
        type=float,
        default=_DEFAULT_GAIN_BGR[2],
        help="gain for red channel",
    )
    parser.add_argument(
        "--alpha",
        required=False,
        type=float,
        default=_DEFAULT_ALPHA_LUT,
        help="alpha for partial histogram equalization",
    )
    args = parser.parse_args()

    enhance(args.input, args.output, (args.gain_b, args.gain_g, args.gain_r), args.alpha)
