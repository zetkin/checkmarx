
import numpy as np
from pyzbar.locations import Point

import scanner


def test_rotation():
    a = Point(0, 1)
    b = Point(2, 0)
    response = scanner.rotation(a, b)
    expected = 2.0 / np.sqrt(5.0)
    np.testing.assert_equal(expected, response)
