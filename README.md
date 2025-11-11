## 1. Quick test of FastAPI

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
1. http://127.0.0.1:8000/docs

## 2. Get Iris predictions using FastAPI
See `README.md` in `./app-iris` folder

## 3. Get Iris predictions using a bash terminal
See `README.md` in `./app-iris-repl` folder