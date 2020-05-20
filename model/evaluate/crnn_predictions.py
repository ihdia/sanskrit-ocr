import json
import os 
import sys

initial_step = int(sys.argv[1])
final_step = int(sys.argv[2])
steps_per_checkpoint = int(sys.argv[3])

if os.path.exists("./model/CRNN/logs/val_preds.txt"):
    os.remove("./model/CRNN/logs/val_preds.txt")
while initial_step<=final_step:

    with open("./model/CRNN/logs/val_preds.txt","a+") as f2:
        f2.write("Step: "+str(initial_step))
        f2.write("\n")

    os.system("CUDA_VISIBLE_DEVICES=0 python ./model/CRNN/tools/test_shadownet.py --filename validating.tfrecords --weights_path model/CRNN/modelss/shadownet_-"+ str(initial_step) + " >> model/CRNN/logs/val_preds.txt") 
    initial_step = initial_step+steps_per_checkpoint