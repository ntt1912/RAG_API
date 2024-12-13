# Langchain Services

## 1. Setup

### 1.1. Donwload data

Require **wget** and **gdown** package

```bash
$ pip3 install wget gdown
$ cd data_source/generative_ai && python download.py
$ cd ../machine_learning && python download.py
```

### 1.2. Run service

```bash
$ pip3 install -r dev_requirements.txt
$ uvicorn src.app:app --host "0.0.0.0" --port 5000 --reload
```
Wait a munite for handling data and starting server.

## 2. Use

Use postman to test API if langserver doesn't work. 

