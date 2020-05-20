import os
import sys
import shutil
import tensorflow as tf
if len(sys.argv)<3:
    sys.exit("Format python model/CRNN/train.py <testing tfrecord filename> <weights_path>")

fil = sys.argv[1]
weights_path = sys.argv[2]


ind = weights_path.find('-')
index = weights_path[weights_path.find('-')+1:]

print("Started Testing")

if os.path.exists(os.getcwd()+"/"+"model/CRNN/logs/test_preds.txt"):
    os.remove(os.getcwd()+"/"+"model/CRNN/logs/test_preds.txt")
os.system("CUDA_VISIBLE_DEVICES=0 python model/CRNN/tools/test_shadownet.py --filename "+fil+" --weights_path "+weights_path + " >> model/CRNN/logs/test_preds.txt") 

