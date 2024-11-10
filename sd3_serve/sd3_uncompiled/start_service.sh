# pip install -r requirements.txt
# pip install -r requirements_service.txt
# git clone https://github.com/pytorch/serve.git
# cd serve && python ./ts_scripts/install_dependencies.py --force --cuda "cu121"
# cd ..
# rm -rf serve

# python download_sd3.py

# cd sd3_model
# zip -r ../sd3_uncompiled/sd3_model.zip *
# cd ../sd3_uncompiled

# mkdir -p model_store
cd sd3_uncompiled

# torch-model-archiver --model-name sd3-model -f --version 1.02 --handler sd3_handler.py --extra-files "sd3_model.zip" -r ../requirements.txt --export-path model_store 
torchserve --start --ts-config config.properties --disable-token-auth  --enable-model-api
