import os

import pytest
from starlette.status import HTTP_400_BAD_REQUEST
from starlette.testclient import TestClient

from checkmarx import main


@pytest.fixture(scope="module")
def client():
    return TestClient(main.APP)


def submit_image(client, relpath):
    """Submit an image to the ``/scan`` endpoint and return the response."""
    dirname = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    test_img = os.path.join(dirname, relpath)
    with open(test_img, "rb") as f:
        return client.post("/scan", files={"image": f})


def test_post_real_questionnaire(client):
    # TODO: Update with appropriate config when implemented.
    response = submit_image(client, "static/img/2019-11-09 12.37.44.jpg")
    assert response.json() == {
        "result": [
            "Header",
            "17 Feb 20:00-21:30",
            "24 Feb 16:00-18:00",
            "25 Feb 12:00-13:30",
        ]
    }


def test_post_unreal_questionnaire(client):
    response = submit_image(client, "static/img/logo.png")
    assert response.status_code == HTTP_400_BAD_REQUEST
