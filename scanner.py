"""
Notes:
    * An A4 page is 210mm x 297mm.
"""

from time import sleep

import cv2 as cv
import numpy as np
from pyzbar import pyzbar
from pyzbar.locations import Point

A4_SIZE = (210, 297)

class DocConfig:
    """A document config will contain all information pertaining to a
    checkbox-style questionnaire.

    All sizes have the format (width, height) and units of mm.

    Attributes:
        page_size: Size of the document.
        checkbox_size: Size of a checkbox in the document.
        qr_size: Size of the QR code.
        qr_offset: Offset from the top-right corner of the QR code to the
          top-right corner of the document.
        x_scale: Document width / QR code width.
        y_scale: Document height / QR code height.
        fields: Questions in the questionnaire.
    """
    page_size = A4_SIZE
    checkbox_size = (12, 10)
    qr_size = (24, 24)
    qr_offset = (20, 20)
    fields = (
        "Is this a questionnaire?",
        "The seminar does a good job integrating.",
        "I made new professional contacts.",
        "One final question."
    )


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


def edge_detect(img):
    """Return a single channel image depicting edged as high-intensity areas."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (15, 15), 0)
    return cv.Canny(gray, 75, 200)


def locate_document_contour(img):
    """Find a document-like contour in an edged image."""
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


def get_single_qr_code(img):
    qr_codes = pyzbar.decode(img)
    if len(qr_codes) == 0:
        print("No QR codes found")
        exit(1)
    if len(qr_codes) > 1:
        print("Too many QR codes found")
        exit(1)
    return qr_codes[0]


def euclidean_distance(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


def rotation(a: Point, b: Point) -> float:
    """Get the rotation from horizontal for the line a->b. In units of
    cos^{-1}."""
    l = euclidean_distance(a, b)
    x = b.x - a.x
    return x / l


def locate_document_contour_qr(image, config):
    """Based on finding a single QR code in the image, return estimated
    document coordinates based on the factors {X,Y}_{SCALE,BUFFER}.

    Notes:
        This entire function currently relies on the document being in
        portrait mode! (According to the dimensions specified in config).

    The coordinates are returned as a tuple in the form:
        (top left, bottom left, bottom right, top right)
    """
    # Get QR code polygon coordinates
    qr_coords = get_single_qr_code(image).polygon

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

    # TODO: This could be optimised and a rotation matrix used, it would
    # require a matrix library in the target language however.

    x_scale = config.page_size[0] / config.qr_size[0]
    y_scale = config.page_size[1] / config.qr_size[1]
    x_offset = config.qr_offset[0] / config.qr_size[0]
    y_offset = config.qr_offset[1] / config.qr_size[1]

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
    """
    Args:
        doc_width: document width in pixels
    """
    blocksize = int(doc_width / 80)
    return max(blocksize + blocksize % 2 - 1, 1)  # Odd and at least 1


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--image", required=True, help="input image")
    parser.add_argument("--debug", action="store_true", help="perform extra debug steps")
    args = parser.parse_args()

    image = cv.imread(args.image)
    config = DocConfig  # TODO: Test others

    # TODO: Use `locate_document_contour` to try and support the estimated
    # contour from QR contour estimator
    contour = locate_document_contour_qr(image, config)
    # contour = locate_document_contour(image)

    contour_np = np.array(contour).astype("int32")

    if args.debug:
        draw_contour(image, contour_np.reshape(4, 1, 2))

    warped = four_point_transform(image, contour_np)

    if args.debug:
        cv.imshow("A", warped)
        wait()

    # Convert the warped image to grayscale, then threshold it
    blocksize = calculate_filter_blocksize(warped.shape[1])
    gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    clipped = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blocksize, 12
    )

    if args.debug:
        cv.imshow("A", clipped)
        wait()


if __name__ == "__main__":
    main()
