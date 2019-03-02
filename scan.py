
from argparse import ArgumentParser
from time import sleep

import cv2 as cv
import numpy as np


parser = ArgumentParser()
parser.add_argument("--image", required=True, help="input image")
parser.add_argument("--debug", action="store_true", help="perform extra debug steps")
args = parser.parse_args()


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


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

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


if __name__ == "__main__":
    image = cv.imread(args.image)
    orig = image.copy()
    edged = edge_detect(image)

    if args.debug:
        cv.imshow("A", edged)
        wait()
        exit()

    contour = locate_document_contour(edged)
    warped = four_point_transform(orig, contour.reshape(4, 2))

    # convert the warped image to grayscale, then threshold it
    warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    _, warped = cv.threshold(warped, 200, 0, cv.THRESH_TRUNC)

    # show the original and scanned images
    cv.imshow("Original", orig)
    cv.imshow("Scanned", warped)
    wait()
