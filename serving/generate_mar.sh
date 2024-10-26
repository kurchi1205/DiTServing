python download_model.py
cd DitModel
zip -r ../model.zip *
cd ..
torch-model-archiver --model-name dit --version 1.0 --handler dit_handler.py --extra-files model.zip -r requirements.txt
torchserve --start --ts-config config.properties --disable-token-auth  --enable-model-api
