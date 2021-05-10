import json
import os
import sys

tfrecordsFile = sys.argv[1]
initial_step = int(sys.argv[2])
final_step = int(sys.argv[3])
steps_per_checkpoint = int(sys.argv[4])
outFile = sys.argv[5]
outFile = './model/CRNN/logs/'+outFile
if os.path.exists(outFile):
    os.remove(outFile)
while initial_step<=final_step:
    print('Starting Checkpoint :',initial_step)
    with open(outFile,"a+") as f2:
        f2.write("Step: "+str(initial_step))
        f2.write("\n")

    os.system("CUDA_VISIBLE_DEVICES=1 python ./model/CRNN/tools/test_shadownet.py --filename "+tfrecordsFile+" --weights_path model/CRNN/c3_modelss/shadownet_-"+ str(initial_step) + " >> " + outFile)
    initial_step = initial_step+steps_per_checkpoint
