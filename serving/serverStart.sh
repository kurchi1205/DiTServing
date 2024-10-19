cp src_infer/* serving/
cp pipelines serving/pipelines

uvicorn app:app --reload --host 127.0.0.1 --port 8000


