import argparse
import io
import os
import tempfile
from pprint import pprint
from typing import List

import fastapi
import pydantic
from PIL import Image, UnidentifiedImageError
from starlette.status import HTTP_400_BAD_REQUEST

from checkmarx import scanner
from checkmarx.exceptions import QRNotFound


APP = fastapi.FastAPI()


class ScanResponse(pydantic.BaseModel):
    checked_boxes: List[str]


@APP.post("/scan", response_model=ScanResponse)
async def scan(image: fastapi.UploadFile = fastapi.File(...)):
    """Scan a document."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, os.path.basename(image.filename)) + ".png"

        # TODO: This is slow! We only need to convert to PNG because the qr
        # scanner does not execute correctly for JPEG...
        Image.open(io.BytesIO(await image.read())).save(path)

        try:
            result = scanner.main(path, debug=False)
        except UnidentifiedImageError as e:
            msg = "Unable to open image file: " + str(e)
            raise fastapi.HTTPException(HTTP_400_BAD_REQUEST,msg) from e
        except QRNotFound as e:
            msg =  "Did not find QR code"
            raise fastapi.HTTPException(HTTP_400_BAD_REQUEST,msg) from e
    return {"checked_boxes": result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument(
        "--debug", action="store_true", help="perform extra debug steps"
    )
    args = parser.parse_args()
    pprint(scanner.main(args.image, args.debug))


if __name__ == "__main__":
    main()
