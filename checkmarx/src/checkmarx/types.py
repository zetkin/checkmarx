from collections import namedtuple


class Point(namedtuple("Point", ["x", "y"])):
    def __truediv__(self, other):
        other = _convert_other(other)
        return Point(self.x / other.x, self.y / other.y)

    def __mul__(self, other):
        other = _convert_other(other)
        return Point(self.x * other.x, self.y * other.y)

    def __sub__(self, other):
        other = _convert_other(other)
        return Point(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        other = _convert_other(other)
        return Point(self.x + other.x, self.y + other.y)


def _convert_other(obj):
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        return Point(obj[0], obj[1])
    if isinstance(obj, (int, float)):
        return Point(obj, obj)
    return obj


Polygon = namedtuple("Polygon", ["topleft", "topright", "bottomright", "bottomleft"])
QR = namedtuple("QR", ["data", "polygon"])
