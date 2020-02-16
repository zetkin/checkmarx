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
been marxed and their corresponding questions.


Requirements
------------

This project relies heavily on a good QR code detector. The polygon output of
the QR code is used to infer the document's coordinates, which in turn is used
to infer checkbox sizes (in pixels, the sizes in mm must be defined).

At the moment `pyzbar` is used (which uses the `zbar` project). `zbar` seems a
bit unreliable, especially in cases where the QR code has been rotated. It also
does not output detailed information about the orientation of the QR code,
which could be very useful to speed up processing time.

Library requirements:
* Python 3
* `zbar` (install using your package manager of choice)
* Python requirements in `setup.py`


Usage
-----

Simply: `checkmarx [-h] --image IMAGE [--debug]`

If the `--debug` flag is used, extra information will be visualised during
the processing / inference stages.

Using the example image `forms/questionnaire-filled.png` should reproduce the
following results.


TODO / Future Ideas
-------------------

* Library userfruct: Book spaces / resources for activities
* Optionally constraint satisfaction for reserving/booking rooms resources
  depending on number of people / alternative times (look at Minizinc)


Results
-------

Inferred document shape from QR code:

![Document](static/img/whole.png)

Discovered checkbox:

![Document](static/img/checkbox.png)

Result:
```
[ ] Is this a questionnaire?
[x] The seminar does a good job integrating.
[x] I made new professional contacts.
[ ] One final question.
```

Implementation Details
----------------------

### Processing Flow

The entire processing flow occurs as follows:
  1. Find a QR code using `pyzbar`
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
```


Further Resources
-----------------

https://docs.opencv.org/3.4/da/d6e/tutorial_py_geometric_transformations.html

https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
