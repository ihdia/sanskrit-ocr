import os
import sys

initial_step = int(sys.argv[1])
final_step = int(sys.argv[2])
steps_per_checkpoint = int(sys.argv[3])
if os.path.exists("./model/attention-lstm/logs/val_preds.txt"):
    os.remove("./model/attention-lstm/logs/val_preds.txt")

while initial_step<=final_step:
    with open('./modelss/checkpoint', 'r') as f:
        # read a list of lines into data
        data = f.readlines()

    get_check = data[0].find('-')
    ckp = data[0][get_check+1:len(data[0])-2]
    print(ckp)


    data[0] = data[0].replace(ckp, str(initial_step))
    print(data[0])
    
    with open('./model/modelss/checkpoint', 'w') as file:
        file.writelines( data )
    """ Change the path accordingly """
    
    with open("./model/attention-lstm/logs/val_preds.txt","a+") as f2:
        f2.write("Step: "+str(initial_step))
        f2.write("\n")

    os.system("CUDA_VISIBLE_DEVICES=1 aocr test ./datasets/validating.tfrecords --max-width 3200 --max-height 200 --max-prediction 600 --gpu-id 1 --model-dir ./modelss --file-id 0") 
    initial_step = initial_step+steps_per_checkpoint
