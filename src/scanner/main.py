import argparse
import tempfile
import sys
from pprint import pprint

import fastapi

from scanner import scanner


APP = fastapi.FastAPI()


@APP.post("/scan")
async def scan(image: fastapi.UploadFile = fastapi.File(...), debug: bool = False):
    """Scan a document."""
    image = await image.read()
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(image)
        result = scanner.main(tmp.name, debug)
    return {"result": result}


def main():
    # TODO: Test other doc configs

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--debug", action="store_true", help="perform extra debug steps"
    )
    args = parser.parse_args()
    pprint(scanner.main(args.image, args.debug))


if __name__ == "__main__":
    main()
