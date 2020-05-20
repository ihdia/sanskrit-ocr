import os
import sys
import shutil
import tensorflow as tf
if len(sys.argv)<5:
    sys.exit("Format python model/CRNN/train.py <training tfrecord filename> <no.of epochs> <weights_path> <steps_per_checkpoint>")

fil = sys.argv[1]
epochs=int(sys.argv[2])
weights_path = sys.argv[3]
steps_per_checkpoint = sys.argv[4]
c=0
for fn in ["model/CRNN/data/tfReal/"+fil]:
    for record in tf.python_io.tf_record_iterator(fn):
        c += 1
print(c)

if weights_path=="0":
    if steps_per_checkpoint=="0":
        print("Started Training")
        os.system("CUDA_VISIBLE_DEVICES=0 python model/CRNN/tools/train_shadownet.py --filename "+fil+" --train_epochs "+str(int(c/32)))
        print("1 Epoch completed")
        for i in range(epochs-1):
            ckpt = tf.train.get_checkpoint_state("model/CRNN/model/shadownet")
            os.system("CUDA_VISIBLE_DEVICES=0 python model/CRNN/tools/train_shadownet.py --filename "+fil+" --train_epochs "+str(int(c/32))+ " --weights_path "+ckpt.model_checkpoint_path)
            print(str(i+2)+" Epochs completed")
    else:
        print("Started Training")
        os.system("CUDA_VISIBLE_DEVICES=0 python model/CRNN/tools/train_shadownet.py --filename "+fil+" --train_epochs "+str(int(c/32))+ " --steps_per_checkpoint "+steps_per_checkpoint)
        print("1 Epoch completed")
        for i in range(epochs-1):
            ckpt = tf.train.get_checkpoint_state("model/CRNN/model/shadownet")
            os.system("CUDA_VISIBLE_DEVICES=0 python model/CRNN/tools/train_shadownet.py --filename "+fil+" --train_epochs "+str(int(c/32))+ " --weights_path "+ckpt.model_checkpoint_path + " --steps_per_checkpoint "+steps_per_checkpoint)
            print(str(i+2)+" Epochs completed")

else:
    print("Started Training")
    if steps_per_checkpoint=="0":
        os.system("CUDA_VISIBLE_DEVICES=0 python model/CRNN/tools/train_shadownet.py --filename "+fil+" --train_epochs "+str(int(c/32))+ " --weights_path "+weights_path)
        print("1 Epoch completed")
        for i in range(epochs-1):
            ckpt = tf.train.get_checkpoint_state("model/CRNN/model/shadownet")
            os.system("CUDA_VISIBLE_DEVICES=0 python model/CRNN/tools/train_shadownet.py --filename "+fil+" --train_epochs "+str(int(c/32))+ " --weights_path "+ckpt.model_checkpoint_path)
            print(str(i+2)+" Epochs completed")
    else:
        os.system("CUDA_VISIBLE_DEVICES=0 python model/CRNN/tools/train_shadownet.py --filename "+fil+" --train_epochs "+str(int(c/32))+ " --weights_path "+weights_path+ " --steps_per_checkpoint "+steps_per_checkpoint)
        print("1 Epoch completed")
        for i in range(epochs-1):
            ckpt = tf.train.get_checkpoint_state("model/CRNN/model/shadownet")
            os.system("CUDA_VISIBLE_DEVICES=0 python model/CRNN/tools/train_shadownet.py --filename "+fil+" --train_epochs "+str(int(c/32))+ " --weights_path "+ckpt.model_checkpoint_path+ " --steps_per_checkpoint "+steps_per_checkpoint)
            print(str(i+2)+" Epochs completed")
