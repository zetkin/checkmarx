"""
Notes:
    * An A4 page is 210mm x 297mm.
"""

from time import sleep
from typing import List, Tuple

import cv2 as cv
import numpy as np
from pyzbar import pyzbar
from pyzbar.locations import Point

A4_SIZE = (210, 297)


TEST_CONFIG = {
    # A document config will contain all information pertaining to a
    # checkbox-style questionnaire.
    # All sizes have the format (width, height) and units of mm.
    # Attributes:
    #     page_size: Size of the document.
    #     checkbox_size: Size of a checkbox in the document.
    #     qr_size: Size of the QR code.
    #     qr_offset: Offset from the top-right corner of the QR code to the
    #       top-right corner of the document.
    #     x_scale: Document width / QR code width.
    #     y_scale: Document height / QR code height.
    #     fields: Questions in the questionnaire.

    "page_size": A4_SIZE,
    "checkbox_size": (8.7, 6),
    "qr_size": (24, 24),
    "qr_offset": (14, 14),
    "fields": (
        "Is this a questionnaire?",
        "The seminar does a good job integrating.",
        "I made new professional contacts.",
        "One final question."
    ),
}


def get_single_qr(img):
    """Get a single QR code from an image."""
    qr_codes = pyzbar.decode(img)
    if len(qr_codes) != 1:
        raise RuntimeError(f"Expected 1 QR code, found {len(qr_codes)}")
    return qr_codes[0]


def fetch_config(url: str) -> dict:
    """Given a QR decoded url, return the doc config."""
    return TEST_CONFIG


def wait():
    while cv.waitKey(0) != 27:
        sleep(0.1)
    cv.destroyAllWindows()


def draw_contour(img, contour):
    """Debug function for showing a contour on an image."""
    copy = img.copy()
    cv.drawContours(copy, [contour], -1, (0, 255, 0), 2)
    cv.imshow("Contour", copy)
    wait()


def locate_document_contour(img):
    """Find a document-like contour in an edged image."""
    # TODO: This is currently an unused function but could be sed to support
    # the estimated document contour calculated from the QR coordinates.

    cnts, _ = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:5]  # 5 largest contours

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen

        # TODO: We should prioritise 4 but also accept up to 8. We should construct
        # ...   an algorithm based on the lenght and point-count. We can also assume
        # ...   that the predominant color will be white.
        if len(approx) == 4:
            paper_contour = approx
            break

    return paper_contour


def four_point_transform(image, pts):
    tl, bl, br, tr = pts
    rect = np.array([tl, tr, br, bl]).astype("float32")

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
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def euclidean_distance(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


def rotation(a: Point, b: Point) -> float:
    """Get the rotation from horizontal for the line a->b. In units of
    cos^{-1}."""
    l = euclidean_distance(a, b)
    x = b.x - a.x
    return x / l


def locate_document_contour_qr(
    image,
    qr_coords: List[Point],
    page_size: Tuple[int, int],
    qr_size: Tuple[int, int],
    qr_offset: Tuple[int, int]
):
    """Based on finding a single QR code in the image, return estimated
    document coordinates based on the factors {X,Y}_{SCALE,BUFFER}.

    Notes:
        This entire function currently relies on the document being in
        portrait mode!

    TODO:
        Account for skewness and perspective shift.

    The coordinates are returned as a tuple in the form:
        (top left, bottom left, bottom right, top right)
    """
    phi = rotation(qr_coords[0], qr_coords[3])

    distances = [
        euclidean_distance(a, b)
        for a, b in zip(qr_coords, qr_coords[1:] + [qr_coords[0]])
    ]

    # Get the estimated {vertical,horizontal} lengths of the QR code in pixels.
    # Given that the QR decoder outputs coordinates in a consistent positional
    # order, we can compute the means of parallel edges.
    qr_y = np.mean(distances[::2])
    qr_x = np.mean(distances[1::2])

    # This could be optimised and a rotation matrix used, it would
    # require a matrix library in the target language however.

    x_scale = page_size[0] / qr_size[0]
    y_scale = page_size[1] / qr_size[1]
    x_offset = qr_offset[0] / qr_size[0]
    y_offset = qr_offset[1] / qr_size[1]

    x0 = qr_coords[0].x - phi * qr_x * (x_scale - (1 + x_offset))
    x1 = qr_coords[1].x - phi * qr_x * (x_scale - (1 + x_offset))
    x2 = qr_coords[2].x + phi * qr_x * x_offset
    x3 = qr_coords[3].x + phi * qr_x * x_offset

    y0 = qr_coords[0].y - phi * qr_y * y_offset
    y1 = qr_coords[1].y + phi * qr_y * (y_scale - (1 + y_offset))
    y2 = qr_coords[2].y + phi * qr_y * (y_scale - (1 + y_offset))
    y3 = qr_coords[3].y - phi * qr_y * y_offset

    return ((x0, y0), (x1, y1), (x2, y2), (x3, y3))


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
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
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


def _n_countour_edges(contour):
    perimeter = cv.arcLength(contour, closed=True)
    return len(cv.approxPolyDP(contour, 0.02 * perimeter, closed=True))


def locate_rectangles(img, area: Tuple[int, int], threshold=0.05):
    """Locate all rectangles in an image that have an area which falls
    within +/- :arg:`threshold` percent of :arg:`area`."""
    cnts, _ = cv.findContours(img.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # Filter by area
    max_area = area * (1 + threshold)
    min_area = area * (1 - threshold)
    cnts = [c for c in cnts if (min_area < cv.contourArea(c) < max_area)]

    # Filter four-sided countours (VERY approximate)
    cnts = [c for c in cnts if _n_countour_edges(c) < 10]

    return cnts


def get_checkboxes(img, page_size, checkbox_size):
    """Return a list of checkboxes, ordered by vertical location in ``img``"""
    # Get checkbox area in pixels^2
    sf = img.shape[0] / page_size[1]  # px/mm
    area_px = np.prod(checkbox_size) * sf**2
    boxes = locate_rectangles(img, area_px)
    boxes = sorted(boxes, key=lambda x: centrepoint(x)[1])
    return [
        shrink_countour(b, 0.8, img.shape[::-1]).round().astype(np.int64)
        for b in boxes
    ]


def percentage_colored(img, contour):
    """Return the percemtage of contour within a binary image which is colored."""
    mask = np.zeros(img.shape, dtype="uint8")
    cv.drawContours(mask, [contour], -1, 1, -1)  # Draw filled contour on mask
    mask = cv.bitwise_and(img, mask)
    # NB sometimes values over 1 are returned, this is a result of how the
    # contour is drawn on the mask, the area of the contour is calculated as
    # smaller than the absolute number of pixels in the mask.
    return 1 - min(mask.sum() / cv.contourArea(contour), 1.0)


def checked_contours(img, contours, threshold=0.07):
    """Find rectangles which have been checked.

    Args:
        img: Image. This should be clipped to {0,1} values.
        contours: Contours to extract.
        threshold: Percentage of colored pixels to determine whether a check
          box is colored. E.g. 0.07 -> if more than 7% of the check box is
          colored, the box is considered checked.
    """
    color = [percentage_colored(img, c) for c in contours]
    return [c > threshold for c in color]


def main():
    # TODO: Test other doc configs

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--image", required=True, help="input image")
    parser.add_argument("--debug", action="store_true", help="perform extra debug steps")
    args = parser.parse_args()

    image = cv.imread(args.image)
    qr_obj = get_single_qr(image)
    config = fetch_config(qr_obj.data.decode())

    # Draw a circle on image
    # im = cv.circle(image, (614, 309), 3, (0,0,0), -1)
    # cv.imwrite("forms/questionnaire3.png", im)
    # exit()

    contour = locate_document_contour_qr(
        image,
        qr_obj.polygon,
        config["page_size"],
        config["qr_size"],
        config["qr_offset"]
    )
    contour_np = np.array(contour).astype("int32")

    if args.debug:
        draw_contour(image, contour_np.reshape(4, 1, 2))

    warped = four_point_transform(image, contour_np)
    if args.debug:
        cv.imshow("A", warped)
        wait()

    clipped = clip_image(warped)
    if args.debug:
        cv.imshow("A", clipped)
        wait()

    checkboxes = get_checkboxes(clipped, config["page_size"], config["checkbox_size"])

    if args.debug:
        for box in checkboxes:
            draw_contour(warped, box)

    checked = checked_contours(clipped, checkboxes)
    for check, field in zip(checked, config["fields"]):
        mark = "x" if check else " "
        print(f"[{mark}] {field}")


if __name__ == "__main__":
    main()
