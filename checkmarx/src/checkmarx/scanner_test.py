
import numpy as np

import scanner
from checkmarx.types import Point


def test_rotation():
    a = Point(0, 1)
    b = Point(2, 0)
    response = scanner.rotation(a, b)
    expected = 2.0 / np.sqrt(5.0)
    np.testing.assert_equal(expected, response)


def test_get_angle():
    topleft = Point(1.0, 1.0)
    topright = Point(2.0, 0.0)
    response = scanner.get_angle(topleft, topright)
    np.testing.assert_close(response, 45)


def main():
    test_rotation()
    test_get_angle()


if __name__ == "__main__":
    main()
