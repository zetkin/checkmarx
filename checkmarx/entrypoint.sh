#!/bin/env bash

APP=checkmarx.main:APP

if [ "$DEPLOYMENT_ENV" = "dev" ]; then 
  pip install pudb
  uvicorn --timeout 999 --reload $APP
else
  gunicorn --threads 1 -b 0.0.0.0:5000 -w 1 --timeout 9999 -k uvicorn.workers.UvicornWorker $APP
fi


