import cv2 as cv
import numpy as np
from utils import float32_to_uint8, uint8_to_float32


def get_clicked_points_from_user(
    image1: np.ndarray, image2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    side_by_side = np.hstack((image1, image2))
    points1, points2 = [], []

    window_name = ("Click on corresponding points in both images. Press "
                   "'enter' when done.")
    cv.namedWindow(window_name)
    print(
        "Click on a point on the left, then the corresponding point on the "
        "right. "
        "Repeat at least 4 times. Press 'enter' when done."
    )

    def draw():
        to_display = side_by_side.copy()
        for i in range(len(points1)):
            cv.circle(to_display, points1[i], 3, (0, 0, 255), -1)
            cv.putText(to_display, str(i), points1[i],
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                       2)
        for i in range(len(points2)):
            cv.circle(to_display, points2[i], 3, (0, 0, 255), -1)
            cv.putText(to_display, str(i), points2[i], cv.FONT_HERSHEY_SIMPLEX,
                       1, (0, 0, 255), 2)
        cv.imshow(window_name, to_display)

    def on_click(event, x, y, flags, param):
        n_left, n_right = len(points1), len(points2)
        if event == cv.EVENT_LBUTTONDOWN:
            if n_left == n_right:
                points1.append((x, y))
            else:
                points2.append((x, y))

    cv.setMouseCallback(window_name, on_click)
    while True:
        draw()
        key = cv.waitKey(1) & 0xFF
        if key == 13:
            cv.destroyAllWindows()
            break

    # Once done, adjust all points to be relative to the top left corner of
    # their respective image
    for i in range(len(points2)):
        points2[i] = (points2[i][0] - image1.shape[1], points2[i][1])

    return np.array(points1), np.array(points2)


def add_weights_channel(image: np.ndarray, border_size: int = 0) -> np.ndarray:
    """Given an image, add a 4th channel to it that contains weights for
    blending. The high-level
    idea is that the weights near the borders of the image should be low to
    reduce visible 'seams'
    where the edge of one image lies near the middle of another.
    """
    h, w = image.shape[:2]
    # Let's say an image is size 200x100. Then, we can't have a 'border' of
    # size bigger than 50
    # because then the size of top+bottom borders would be greater than the
    # height of the image.
    # This line makes sure that the border size is at most half the height
    # and half the width.
    border_size = min(border_size, h // 2, w // 2)

    # The weights_x and weights_y arrays are 1D arrays of size w and h,
    # respectively. They ramp
    # up from 0 to 1 in 'border_size' steps, then stay at 1 for the middle section of the image,
    # then ramp back down to 0 in 'border_size' steps.
    # Graphically, the weights look like this:
    #
    #    /-----------------\
    #   /                   \
    #  /                     \
    #  <-> size of this 'ramp' is 'border_size'
    weights_x = np.hstack(
        [
            np.linspace(0, 1, border_size),
            np.ones(w - 2 * border_size),
            np.linspace(1, 0, border_size),
        ]
    )
    weights_y = np.hstack(
        [
            np.linspace(0, 1, border_size),
            np.ones(h - 2 * border_size),
            np.linspace(1, 0, border_size),
        ]
    )
    weights_xy = np.minimum(weights_y[:, np.newaxis], weights_x[np.newaxis, :]).astype(np.float32)
    return np.dstack((image, weights_xy))


def apply_homography(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Given a 3x3 homography matrix and an 2xn array of 2D points (each point as a column vector),
    return the new 2xn array of points after applying the homography.
    """
    _, n = points.shape
    points_augmented = np.vstack((points, np.ones((1, n))))
    # Matrix multiplication of h with points_augmented. The '@' operator is a built-in python
    # operator which numpy interprets as 'matrix multiplication'.
    new_points = h @ points_augmented
    # new_points_2 = new_points / new_points[2, :]
    return (new_points / new_points[-1, :])[:points.shape[0], :]


def homography_then_translate(homog: np.ndarray, translate: tuple[float, float]) -> np.ndarray:
    """Given a 3x3 homography matrix 'homog' that will map from coordinates x to coordinates y like
    y = homog @ x, return a new homography matrix homog2 such that y2 = homog2 @ x is equivalent to
    y2 = homog @ x + t for the given translation vector t. In other words, this function combines
    "homography followed by translation" into a single homography matrix.
    """
    # Make 'translator' - another 3x3 homography matrix that *just* does translation
    translator = np.eye(3)
    translator[:2, 2] = np.array(translate)
    # The idea is that translator @ (homog @ x) is the same as (homog @ x) + translate, and we can
    # group the parentheses differently to get (translator @ homog) @ x
    return translator @ homog


def calculate_bounding_box_of_warps(homographies, images):
    """Given a list of homography matrices and a list of images, compute the bounding box of the
    warped images by seeing where their corners land after warping.

    To do this, we'll compute the new 2D location of the *corners* of each image. Then, we'll
    compute the bounding box of all of those points.

    Returns the (left, top, width, height) of the bounding box.
    """
    warped_corners = np.hstack(
        [
            apply_homography(
                h,
                np.array(
                    [[0, 0], [im.shape[1], 0], [0, im.shape[0]], [im.shape[1], im.shape[0]]]
                ).T,
            )
            for h, im in zip(homographies, images)
        ]
    )
    # Compute the bounding box that contains all the warped corners
    left = int(np.floor(np.min(warped_corners[0])))
    right = int(np.ceil(np.max(warped_corners[0])))
    top = int(np.floor(np.min(warped_corners[1])))
    bottom = int(np.ceil(np.max(warped_corners[1])))
    w, h = right - left, bottom - top
    return left, top, w, h


def weighted_blend(images: list[np.ndarray]) -> np.ndarray:
    """Given a list of images each with 4 channels (e.g. BGRW or LABW), blend them together using
    the weights in the 4th channel. Return the blended image. In the formulas below, image[i] refers
    to just the first three channels of the i'th image (i.e. BGR or LAB or whatever), and w[i]
    refers to the value in the 4th 'weight' channel of the image.

    Each output pixel will be a weighted average of the two corresponding input pixels, so

        output = sum(w[i] * image[i]) / sum(w)

    Anywhere that sum(w) is zero, the outputs will be zero.
    """
    weights = [im[:, :, 3:] for im in images]
    images = [im[:, :, :3] for im in images]
    sum_of_weights = np.sum(weights, axis=0)
    sum_of_weights[sum_of_weights == 0] = np.inf  # Avoid divide by zero
    return np.sum([w * im for w, im in zip(weights, images)], axis=0) / sum_of_weights


def stitch(
    images: list[np.ndarray],
    points: list[np.ndarray],
    reference_image_index: int,
    border_blend_size: int,
) -> np.ndarray:
    """This function puts all the pieces together and returns a composited panorama image. The
    'images' list contains the images to stitch together, and the 'points' list contains the
    corresponding points in each image. The 'reference_image_index' is the index of the image
    that will be used as the reference coordinate system for the output panorama. The
    'border_blend_size' is the size of the border blending region in pixels.
    """
    if not all(len(p) == len(points[0]) for p in points):
        raise ValueError("Number of points in each image must be equal (corresponding points)")

    # Convert images to floating point representation so that we don't have to worry about overflow
    # or underflow while doing math.
    images = [uint8_to_float32(im) for im in images]

    # Create the 'weights' for each pixel in each image as a 4th channel.
    images = [add_weights_channel(im, border_blend_size) for im in images]

    # Compute 3x3 homography matrix that maps from images[j]'s coordinate system to image[ref]'s
    # coordinate system where image[ref] is the reference image and images[j] is any other image.
    homographies = [
        cv.findHomography(points[j], points[reference_image_index])[0] for j in range(len(
            images))
    ]

    # Next, we need to calculate how big of an output image we need to hold all the warped images.
    left, top, new_w, new_h = calculate_bounding_box_of_warps(homographies, images)

    # We can't use negative rows/cols to index into images. If 'left' or 'top' is negative, then
    # we need to adjust all the homography matrices so that they add back in a translation of
    # (|left|, |top|). This will shift all the images to the right and/or down so that the top-left
    # corner of the combined image is at (0, 0).
    x_shift, y_shift = max(0, abs(left)), max(0, abs(top))
    homographies = [homography_then_translate(h, (x_shift, y_shift)) for h in homographies]

    # Warp all images (plus their 4th 'weights' channel) into the new coordinate system
    warped = [cv.warpPerspective(im, h, (new_w, new_h)) for h, im in zip(homographies,
                                                                         images)]

    # Blend the warped images together using the weights in the 4th channel (note that the weights
    # were also transformed by the homography)
    stitched = weighted_blend(warped)

    return float32_to_uint8(stitched)


def main(image1: str, image2: str, border_size: int, output: str):
    # Read in the images and convert to float32
    im1 = cv.imread(image1)
    im2 = cv.imread(image2)

    # NOTE – you may modify the 'main' function any way you like. You might therefore find it useful
    # to hard-code a 'points1' and 'points2' array while debugging everything else. That way, you
    # won't have to do the point-and-click interface for each debug iteration.
    points1, points2 = get_clicked_points_from_user(im1, im2)

    panorama = stitch([im1, im2], [points1, points2], 0, border_size)

    cv.imwrite(output, panorama)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image1", required=True, help="Path to the first image")
    parser.add_argument("--image2", required=True, help="Path to the second image")
    parser.add_argument("--border-size", type=int, default=50, help="Size of the border blend")
    parser.add_argument("--output", required=True, help="Path to save the combined image")
    args = parser.parse_args()

    main(**vars(args))
