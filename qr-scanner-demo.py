
from argparse import ArgumentParser
from time import sleep

import cv2 as cv
from pyzbar import pyzbar

parser = ArgumentParser()
parser.add_argument("--image", required=True, help="input image")
parser.add_argument("--debug", action="store_true", help="perform extra debug steps")
args = parser.parse_args()


orig = cv.imread(args.image)


def show(img):
    cv.imshow("a", img)
    while cv.waitKey(0) != 27:
        sleep(0.1)
    cv.destroyAllWindows()


# PyZbar
qr_codes = pyzbar.decode(orig)  # Benchmark
image = orig.copy()
colors = [(255,0,0), (0,255,0), (120,155,120), (0,0,255)]
for obj in qr_codes:
    for p, c in zip(obj.polygon, colors):
        cv.circle(image, (p.x, p.y), 1, c, 9)
show(image)


# OpenCV
image = orig.copy()
decoder = cv.QRCodeDetector()
data, bbox, rectifiedImage = decoder.detectAndDecode(orig)
for obj in bbox:
    for p in obj:
        cv.circle(image, tuple(p), 1, (0,255,0), 9)
show(image)
show(rectifiedImage)
