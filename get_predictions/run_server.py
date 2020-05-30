import os
import sys

if len(sys.argv)<2:
    sys.exit("Format python run_server.py <model_type>")

model_type = sys.argv[1]
os.system("tensorflow_model_server --port=9000 --rest_api_port=9001 --model_name=sanskrit --model_base_path="+os.getcwd()+'/'+model_type)
