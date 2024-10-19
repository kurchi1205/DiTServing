cp src_infer/* serving/
cp pipelines serving/pipelines

pip install -r requirements_test.txt
pip install -r requirements_server.txt

uvicorn app:app --reload --host 127.0.0.1 --port 8000


