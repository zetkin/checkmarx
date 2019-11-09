"""
Document Configs
================

A document config will contain all information pertaining to a
checkbox-style questionnaire.

All sizes have the format (width, height) and units of mm.

Attributes:
    page_size: Size of the document.
    checkbox_size: Size of a checkbox in the document.
    qr_size: Size of the QR code.
    qr_offset: Offset from the top-right corner of the QR code to the
      top-right corner of the document.
    x_scale: Document width / QR code width.
    y_scale: Document height / QR code height.
    fields: Questions in the questionnaire.
"""

A4_SIZE = (210, 297)


TEST_CONFIG = {
    "page_size": A4_SIZE,
    "checkbox_size": (8.7, 6),
    "qr_size": (24, 24),
    "qr_offset": (14, 14),
    "fields": (
        "Is this a questionnaire?",
        "The seminar does a good job integrating.",
        "I made new professional contacts.",
        "One final question."
    ),
}


FEMINISTISKA_CONFIG = {
    "page_size": A4_SIZE,
    "checkbox_size": (13, 9),
    "qr_size": (28, 27),
    "qr_offset": (11, 10),
    "fields": (
        "Header",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
    ),
}
