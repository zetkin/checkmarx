"""
Document Configs
================

A document config will contain all information pertaining to a
checkbox-style questionnaire.

All sizes have the format (width, height) and units of mm.
"""
from typing import List

import pydantic


class Config(pydantic.BaseModel):
    """
    Attributes:
        page_size: Size of the document.
        checkbox_size: Size of a checkbox in the document.
        qr_size: Size of the QR code.
        qr_offset: Offset from the top-right corner of the QR code to the
          top-right corner of the document.
        fields: Questions in the questionnaire.
    """

    page_size: List[int]
    checkbox_size: List[int]
    qr_size: List[int]
    qr_offset: List[int]
    checkbox_titles: List[List[str]]


A4_SIZE = [210, 297]


FEMINISTISKA_CONFIG = Config.parse_obj(
    {
        "page_size": A4_SIZE,
        "checkbox_size": [13, 9],
        "qr_size": [28, 27],
        "qr_offset": [11, 10],
        "checkbox_titles": [
            ["Header"],
            ["15 Feb 16:00-18:00"],
            ["2"],
            ["3"],
            ["4"],
            ["5"],
            ["6"],
            ["7"],
            ["8"],
            ["9"],
            ["10"],
            ["11"],
            ["12"],
            ["13"],
            ["14"],
            ["15"],
            ["16"],
            ["17"],
            ["18"],
            ["19"],
            ["20"],
        ],
    }
)
