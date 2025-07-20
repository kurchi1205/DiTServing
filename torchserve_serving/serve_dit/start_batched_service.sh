# pip install -r requirements.txt
# git clone https://github.com/pytorch/serve.git
# cd serve && python ./ts_scripts/install_dependencies.py --force --cuda "cu121"
# cd ..
# rm -rf serve
cp ../pipelines/base_pipeline_dit.py base_pipeline_dit.py
python download_model.py

cd Dit_model
zip -r ../model.zip *
cd ..

mkdir -p model_store

torch-model-archiver --model-name dit-model -f --version 1.0 --handler dit_handler.py --extra-files "model.zip,base_pipeline_dit.py" -r requirements.txt --export-path model_store 
torchserve --start --ts-config config_batched.properties --disable-token-auth  --enable-model-api
rm -rf base_pipeline_dit.py