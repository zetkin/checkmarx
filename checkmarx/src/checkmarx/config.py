"""
Document Configs
================

A document config will contain all information pertaining to a
checkbox-style questionnaire.
"""
from typing import List

import pydantic

from checkmarx.types import Point


class DocumentConfig(pydantic.BaseModel):
    """
    All sizes have the format (width, height) and units of mm.

    Attributes:
        page_size: Size of the document.
        checkbox_size: Size of a checkbox in the document.
        qr_size: Size of the QR code.
        qr_offset: Offset from the top-left corner of the document to the
          top-left corner of the QR code.
        fields: Questions in the questionnaire.
    """

    page_size: Point
    checkbox_size: Point
    qr_size: Point
    qr_offset: Point
    checkbox_titles: List[List[str]]

    @pydantic.validator("*", pre=True)
    def convert_to_named_tuple(cls, value, field):
        if field.type_ is Point:
            return Point(*value)
        return value
