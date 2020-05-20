import os
os.system("tensorflow_model_server --port=9000 --rest_api_port=9001 --model_name=sanskrit --model_base_path="+os.getcwd()+'/exported-model')
