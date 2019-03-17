
from time import sleep
import cv2 as cv

warped = cv.imread("forms/feminism.jpg")
gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
clipped = cv.adaptiveThreshold(
    gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 12
)


def wait():
    while cv.waitKey(0) != 27:
        sleep(0.1)
    cv.destroyAllWindows()


cv.imshow("Original", clipped)
wait()
