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
            ["15 Feb 20:00-21:30"],
            ["17 Feb 11:00-13:30"],
            ["17 Feb 16:00-18:00"],
            ["17 Feb 20:00-21:30"],
            ["19 Feb 11:00-13:30"],
            ["19 Feb 16:00-18:00"],
            ["21 Feb 20:00-21:30"],
            ["22 Feb 07:00-09:00"],
            ["22 Feb 16:00-18:00"],
            ["22 Feb 20:00-21:30"],
            ["24 Feb 11:00-13:30"],
            ["24 Feb 16:00-18:00"],
            ["24 Feb 17:00-18:00"],
            ["24 Feb 20:00-21:30"],
            ["25 Feb 12:00-13:30"],
            ["25 Feb 17:00-18:00"],
            ["26 Feb 07:00-09:00"],
            ["26 Feb 11:00-13:30"],
            ["26 Feb 16:00-18:00"],
        ],
    }
)
