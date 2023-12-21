FROM python:3.10.8

WORKDIR /app

COPY . /app

RUN pip install dvc[s3]

RUN dvc init --no-scm -f
RUN dvc remote add -d myremote s3://dvc-art
RUN dvc remote modify myremote endpointurl http://minio:9000
RUN dvc remote modify myremote access_key_id minioadmin
RUN dvc remote modify myremote secret_access_key minioadmin
RUN dvc remote modify myremote region eu-west-1

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app:app"]
