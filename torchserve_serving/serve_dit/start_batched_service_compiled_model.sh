cd Dit_model
zip -r ../model.zip *
cd ..

mkdir -p model_store

torch-model-archiver --model-name dit-model-batched-compiled -f --version 1.0 --handler dit_compiled_batch_handler.py --extra-files "model.zip,base_pipeline_dit.py,dit_handler.py" -r requirements.txt --export-path model_store 
torchserve --start --ts-config config_compiled_batched.properties --disable-token-auth  --enable-model-api
