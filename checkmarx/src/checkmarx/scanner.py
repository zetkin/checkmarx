import re

import cv2 as cv
import numpy as np
import requests
from PIL import Image

from checkmarx import utils
from checkmarx.config import DocumentConfig
from checkmarx.exceptions import QRNotFound
from checkmarx.types import Point, Polygon, QR

CORNER_PATTERN = "".join(r" (\((?P<x%s>\d+),(?P<y%s>\d+)\))" % (i, i) for i in range(4))
QUIRC_RE = re.compile(
    r"corners:(?P<corners>%s).+Payload: (?P<data>.+)" % CORNER_PATTERN,
    flags=re.DOTALL,
)


def get_single_qr(img_path):
    output = utils.exe("./qrtest", ("-v", "-d", img_path))
    match = QUIRC_RE.search(output)
    # TODO: Raise error if multiple found
    if match:
        corners = [Point(*map(int, match.group(f"x{i}", f"y{i}"))) for i in range(4)]

        # TODO: Replace with real URL when ready
        data = match.group("data")
        data = "http://metadata-server:8000/feminism-handout"

        return QR(data, Polygon(*corners))
    raise QRNotFound


def fetch_config(url):
    """Given a QR decoded url, return the doc config."""
    return DocumentConfig.parse_obj(requests.get(url).json())


def draw_contour(img, contour):
    """Debug function for showing a contour on an image."""
    copy = img.copy()
    cv.drawContours(copy, contour, -1, (0, 255, 0), 2)
    imshow(copy)


def four_point_transform(image, pts):
    tl, tr, br, bl = pts
    rect = pts.astype("float32")

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def locate_document(
    qr_coords: Polygon,
    page_size: Point,
    qr_size: Point,
    qr_offset: Point,
):
    """Based on finding a single QR code in the image, return estimated
    document coordinates.
    """
    i_hat = np.array(qr_coords.topright - qr_coords.topleft) / qr_size.x
    j_hat = np.array(qr_coords.bottomleft - qr_coords.topleft) / qr_size.y
    # `A` is the transformation matrix for change of basis
    A = np.c_[i_hat, j_hat]

    topleft_mm = -np.array(qr_offset)
    topright_mm = topleft_mm + [page_size.x, 0]
    bottomleft_mm = topleft_mm + [0, page_size.y]
    bottomright_mm = topleft_mm + page_size

    topleft = A.dot(topleft_mm) + qr_coords.topleft
    topright = A.dot(topright_mm) + qr_coords.topleft
    bottomleft = A.dot(bottomleft_mm) + qr_coords.topleft
    bottomright = A.dot(bottomright_mm) + qr_coords.topleft

    return (topleft, topright, bottomright, bottomleft)


def calculate_filter_blocksize(doc_width):
    """Dynamically approximate a good blocksize for adaptive thresholding
    based on the size of a document.

    Args:
        doc_width: document width in pixels
    """
    blocksize = int(doc_width / 80)
    return max(blocksize + blocksize % 2 - 1, 1)  # Odd and at least 1


def centrepoint(contour):
    """Given an arbitrary contour, return the centrepoint (x, y)."""
    return contour.mean(axis=(0, 1))


def shrink_countour(contour, shrinkage, shape):
    """Shrink a contour by a given amount.

    Method:
        1. Subtract the centre x/y from the coordinates.
        2. Multiply by shrinkage.
        3. Add centre x/y.
    """
    M = cv.moments(contour)
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    centre = np.array((cx, cy)).reshape(1, 1, -1)
    new_contour = (contour - centre) * shrinkage + centre
    # new_contour[:, 0, 0] = new_contour[:, 0, 0].clip(0, shape[0])
    # new_contour[:, 0, 1] = new_contour[:, 0, 1].clip(0, shape[1])
    return new_contour


def clip_image(img):
    """Convert an image to grayscale, then threshold it."""
    blocksize = calculate_filter_blocksize(img.shape[1])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blocksize, 12
    )


def has_area(contour, area, threshold):
    """Return whether a contour has an area within a given threshold."""
    max_area = area * (1 + threshold)
    min_area = area * (1 - threshold)
    return min_area < cv.contourArea(contour) < max_area


def four_sided(contour):
    """Verrrry approximate function for whether a contour is four-sided."""
    perimeter = cv.arcLength(contour, closed=True)
    return 2 < len(cv.approxPolyDP(contour, 0.02 * perimeter, closed=True)) < 6


def within_aspect_ratio(contour, aspect_ratio, threshold):
    """Return whether a contour has an aspect ratio within a given threshold."""
    _x, _y, w, h = cv.boundingRect(contour)
    return aspect_ratio * (1 - threshold) < w / h < aspect_ratio * (1 + threshold)


def group_columns(boxes, threshold_px):
    """Group boxes by their vertical alignment within a given threshold."""
    pass


def minimum_rows(boxes, minimum):
    """Filter away columns of boxes which have fewer than `minimum` rows."""
    pass


def locate_rectangles(img, area, aspect_ratio, threshold=0.40):
    """Locate all rectangles in an image that have an area which falls
    within +/- :arg:`threshold` percent of :arg:`area`."""
    contours, _ = cv.findContours(img.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    for f in (
        lambda x: has_area(x, area, threshold),
        four_sided,
        lambda x: within_aspect_ratio(x, aspect_ratio, threshold),
    ):
        contours = filter(f, contours)

    return contours


def get_checkboxes(img, page_size, checkbox_size):
    """Return a list of checkboxes, ordered by vertical location in ``img``"""
    # TODO: Perform appropriate column grouping using n-columns from checkbox_titles
    # Get checkbox area in pixels^2
    sf = img.shape[0] / page_size[1]  # px/mm
    area_px = np.prod(checkbox_size) * sf ** 2
    # TODO: Also filter by aspect ratio
    aspect_ratio = checkbox_size[0] / checkbox_size[1]
    boxes = locate_rectangles(img, area_px, aspect_ratio)
    boxes = sorted(boxes, key=lambda x: centrepoint(x)[1])  # Sort by highest on page
    return [
        [shrink_countour(b, 0.9, img.shape[::-1]).round().astype(np.int64)]
        for b in boxes
    ]


def percentage_colored(img, contour):
    """Return the percentage of contour within a binary image which is colored."""
    mask = np.zeros(img.shape, dtype="uint8")
    cv.drawContours(mask, [contour], -1, 1, -1)  # Draw filled contour on mask
    area = (mask > 0).sum()
    extracted = cv.bitwise_and(img, mask)
    # TODO: We could multiply by a gaussian kernel here to give greater weight
    # to the central pixels
    return 1 - extracted.sum() / area


def checked_contours(img, contours, threshold):
    """Find rectangles which have been checked.

    Args:
        img: Image. This should be clipped to {0,1} values.
        contours: Contours to extract.
        threshold: Percentage of colored pixels to determine whether a check
          box is colored. E.g. 0.07 -> if more than 7% of the check box is
          colored, the box is considered checked.
    """
    color = [
        [percentage_colored(img, c) for c in contour_columns]
        for contour_columns in contours
    ]
    return [[c > threshold for c in color_columns] for color_columns in color]


def imshow(img):
    import matplotlib.pyplot as plt

    plt.imshow(img)
    plt.show()


def main(image_path, debug):
    image_pil = Image.open(image_path)
    image = np.array(image_pil)

    qr_obj = get_single_qr(image_path)

    if debug:
        print("QR location:", qr_obj.polygon)
        draw_contour(image, np.array(qr_obj.polygon).astype("int32").reshape(4, 1, 2))

    config = fetch_config(qr_obj.data)

    contour = locate_document(
        qr_obj.polygon,
        config.page_size,
        config.qr_size,
        config.qr_offset,
    )
    contour_np = np.array(contour).astype("int32")
    if debug:
        print("Document location:", contour)
        draw_contour(image, contour_np.reshape(4, 1, 2))

    image = four_point_transform(image, contour_np)
    if debug:
        imshow(image)

    clipped = clip_image(image)
    if debug:
        imshow(clipped)

    checkboxes = get_checkboxes(clipped, config.page_size, config.checkbox_size)
    if debug:
        print(f"Found {len(checkboxes)} check boxes")
        for contour in checkboxes:
            draw_contour(image, contour)

    checked = checked_contours(clipped, checkboxes, threshold=0.01)
    if checked:
        titles = np.array(config.checkbox_titles)
        checked = np.array(checked)
        result = list(titles[: checked.shape[0], :][checked])
    else:
        result = []

    return result
