FROM python:3.7-slim

WORKDIR /usr/src/app

# build-essential is required for gcc and relevant headers
RUN apt update -qq && apt install -yq --no-install-recommends \
      build-essential \
      libglib2.0-0 \
      libzbar-dev

COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
RUN pip install -e .

ENTRYPOINT entrypoint.sh
