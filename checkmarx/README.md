![CheckMarx](static/img/logo.png)

A prototype document scanner for finding checkboxes in questionnaires and
determining which of them have been marxed.

A QR code will be used to define characteristics of the questionnaire (likely
via an online resource rather than directly encoded in the QR code), such as:
  * Document size in mm (e.g. `(210, 297)` for A4)
  * Checkbox sizes in mm
  * QR code size in mm
  * QR code position offset in mm
  * Checkmark titles

This information will be used to locate the checkboxes and determine which have
been marxed and their corresponding fields.


Requirements
------------

This project relies heavily on a good QR code detector. The polygon output of
the QR code is used to infer the document's coordinates, which in turn is used
to infer checkbox sizes (in pixels, the sizes in mm must be defined).

At the moment [`quirc`](https://github.com/dlbeer/quirc/) is used which is
written in C. A pre-compiled program has been committed to this repository
which works in the docker setup defined. If a local setup is required, pull
the `quirc` repo and compile the `qrtest` program then either copy it or link
to it from this directory.


Usage
-----

Simply: `checkmarx [-h] --image IMAGE [--debug]`

If the `--debug` flag is used, extra information will be visualised during
the processing / inference stages.


Implementation Details
----------------------

### Processing Flow

The entire processing flow occurs as follows:
  1. Find a QR code using a system call to `quirc`
  2. Fetch the document config from the QR code message
  3. Infer the document shape (in pixels) based on the size of the QR polygon,
     and the details from the document config
  4. Extract and threshold the document from the whole image to produce a single
     channel binary image
  5. Collect all checkboxes in the document by searching for contours which
     match the stated size of the checkboxes from the config (these are sorted
     by vertical position)
  6. Determine which boxes are marxed based on whether they have over a certain
     percentage of black pixels
  7. Return an array of marxed boxes, sorted in descending order
  8. Lose chains


### QR Code Data

The QR code should encode a URL which can be used to fetch a JSON object
containing all document information:

```json
{
    "page_size": [210, 297],
    "checkbox_size": [12, 10],
    "qr_size": [24, 24],
    "qr_offset": = [14, 14],
    "checkbox_titles": [
        ["Is this a questionnaire?"],
        ["The seminar does a good job integrating."],
        ["I made new professional contacts."],
        ["One final question."]
    ]
}
```


Further Resources
-----------------

https://github.com/dlbeer/quirc/

https://docs.opencv.org/3.4/da/d6e/tutorial_py_geometric_transformations.html

https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
