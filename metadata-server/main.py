import fastapi


APP = fastapi.FastAPI()


CONFIGS = {
    "feminism-handout": {
        "page_size": [210, 297],
        "checkbox_size": [13, 9],
        "qr_size": [28, 27],
        "qr_offset": [171, 10],
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
}


@APP.get("/{id}")
def get_metadata(id: str):
    return CONFIGS.get(id)
