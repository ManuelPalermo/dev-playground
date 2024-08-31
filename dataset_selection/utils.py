import glob
from collections.abc import Sequence
from typing import Any

import cv2
import imutils
import numpy as np
from numpy.typing import NDArray


def draw_color_mask(img, borders, color=(0, 0, 0)):
    """Draws a color mask on the image."""
    h = img.shape[0]
    w = img.shape[1]

    x_min = int(borders[0] * w / 100)
    x_max = w - int(borders[2] * w / 100)
    y_min = int(borders[1] * h / 100)
    y_max = h - int(borders[3] * h / 100)

    img = cv2.rectangle(img, (0, 0), (x_min, h), color, -1)
    img = cv2.rectangle(img, (0, 0), (w, y_min), color, -1)
    img = cv2.rectangle(img, (x_max, 0), (w, h), color, -1)
    img = cv2.rectangle(img, (0, y_max), (w, h), color, -1)

    return img


def preprocess_image_change_detection(img, gaussian_blur_radius_list=None, black_mask=(5, 10, 5, 0)):
    gray = img.copy()
    # gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY) # no need, sa images already come pre-processed now
    if gaussian_blur_radius_list is not None:
        for radius in gaussian_blur_radius_list:
            gray = cv2.GaussianBlur(gray, (radius, radius), 0)

    gray = draw_color_mask(gray, black_mask)

    return gray


def compare_frames_change_detection(prev_frame, next_frame, min_contour_area):
    """Compares two frames and returns the score and the contours."""
    frame_delta = cv2.absdiff(prev_frame, next_frame)
    thresh = cv2.threshold(frame_delta, 45, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    score = 0
    res_cnts = []
    for c in cnts:
        if cv2.contourArea(c) < min_contour_area:
            continue

        res_cnts.append(c)
        score += cv2.contourArea(c)

    return score, res_cnts, thresh


def load_image_raw(path: str):
    """Loads an image from path."""
    img = cv2.imread(path)
    return img


def load_image_preprocess(path: str) -> NDArray[np.uint8] | None:
    """Loads an image from path."""
    img = load_image_raw(path)

    # ignore images which cannot be properly read
    if img is None:
        return None

    proc_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proc_img = cv2.resize(proc_img, (640, 480))
    proc_img = proc_img.astype(np.uint8)
    return proc_img


def visualize_image(
    imgs: Sequence[NDArray[np.uint8]],
    name: str = "Image",
    factor: int = 2,
    wait_ms: int = 0,
):
    """Visualize images."""
    vis_img = np.concatenate(imgs, axis=1)
    new_shape = (int(vis_img.shape[1] / factor), int(vis_img.shape[0] / factor))
    img_small = cv2.resize(vis_img, dsize=new_shape, interpolation=cv2.INTER_AREA)
    cv2.imshow(name, img_small)
    cv2.waitKey(wait_ms)


def draw_countours(img: NDArray[np.uint8], cnts: list[Any]):
    """Draws contours on the image."""
    img_cnts = img.copy()
    img_cnts = cv2.drawContours(img_cnts, cnts, -1, (0, 255, 255), 2)
    return img_cnts


def find_folder_images(path: str):
    """Returns a list with all the images inside folder."""
    img_files = glob.glob(path + "/*.png")
    return img_files


def split_list_in_chunks(lst: Sequence[Any], chunk_size: int | None = None) -> list[Sequence[Any]]:
    """Splits a list into chunks of desired size."""
    if chunk_size is None:
        chunk_size = len(lst)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]
