FROM python:3.7-slim

WORKDIR /usr/src/app

# build-essential is required for gcc and relevant headers
RUN apt update -qq && apt install -yq --no-install-recommends \
      build-essential \
      libglib2.0-0 \
      libzbar-dev

COPY . .
RUN pip install -e .

ENTRYPOINT gunicorn --threads 1 -b 0.0.0.0:5000 -w 1 -k uvicorn.workers.UvicornWorker checkmarx.main:APP
