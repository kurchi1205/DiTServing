pip install -r requirements.txt
pip install -r requirements_service.txt
git clone https://github.com/pytorch/serve.git
cd serve && python ./ts_scripts/install_dependencies.py --force --cuda "cu121"
cd ..
rm -rf serve
python download_model.py
