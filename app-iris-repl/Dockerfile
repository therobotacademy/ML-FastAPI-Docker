FROM python:3.11
WORKDIR /src
COPY requirements.txt /src/requirements.txt
RUN pip install -r requirements.txt
COPY ./src /src

RUN mkdir -p /src/data
COPY data data

CMD ["python", "main.py"]
