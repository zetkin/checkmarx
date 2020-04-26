import argparse
import json

import cv2 as cv


def show_image(path):
    img = cv.imread(path, cv.IMREAD_UNCHANGED)
    cv.imshow(path, img)


def read_clicks(name, n):
    clicks = []

    def fn(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONUP:
            pos = (x, y)
            clicks.append(pos)
            print(f"Click {len(clicks)}:", pos)

    cv.setMouseCallback(name, fn)

    while len(clicks) < n:
        key = cv.waitKey(1)
        if key in (113, 27):
            exit()

    return clicks


def annotation_path(img_path):
    return ".".join(img_path.split(".")[:-1]) + ".json"


def write_annotation(path, annotation):
    with open(path, "wt") as f:
        json.dump(annotation, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="Images to annotate")
    args = parser.parse_args()

    for path in args.images:
        print(f"Annotating '{path}'...")
        show_image(path)
        print("Paper coords")
        paper_coords = read_clicks(path, 4)
        print("QR coords")
        qr_coords = read_clicks(path, 4)
        cv.destroyAllWindows()
        annotation = {
            "path": path,
            "paper_coords": paper_coords,
            "qr_coords": qr_coords,
        }
        write_annotation(annotation_path(path), annotation)


if __name__ == "__main__":
    main()
