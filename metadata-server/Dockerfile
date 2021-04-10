FROM python:3.9-slim

COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
ENTRYPOINT ["uvicorn", "--host", "0.0.0.0", "main:APP"]
