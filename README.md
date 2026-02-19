# Serving a Machine Learning Model

This repository gives you the tools to start with:

* FastAPI
* Docker
* Serve a Machine Learning model via REST API

Folder structure:

```
.
├── app
│   ├── __init__.py
│   └── main.py
├── Dockerfile
├── requirements.txt
└── env
```

---

## 1. Quick test of FastAPI

Run a tiny FastAPI server:

```
cd app
uvicorn main:app
```

---

## 2. Using Docker

Build image:

```bash
docker build -t fastapi_learn .
```

Launch app:

```
docker run -d -p 8000:80 fastapi_learn
```

Visit:

1. http://127.0.0.1:8000/
2. http://127.0.0.1:8000/docs

---

## 3. Serving a Machine Learning model

### Get Iris predictions using FastAPI

See `README.md` in `./app-iris` folder


### Get Iris predictions using a bash terminal

See `README.md` in `./app-iris-repl` folder
