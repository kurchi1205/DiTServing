cp src_infer/* serving/
cp -r pipelines serving/pipelines

cd serving
pip install -r requirements_test.txt
pip install -r requirements_server.txt

uvicorn app:app --reload --host 127.0.0.1 --port 8000

rm infer_*.py
rm requirements_test.txt
rm -rf pipelines


