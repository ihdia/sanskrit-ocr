import numpy as np 
import os
import random
import sys
import math

if len(sys.argv)<4:
   sys.exit("Format: python prep_scripts/train_test_split.py train_split val_split test_split")

train_split, val_split, test_split = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])

with open("label_data/annot_real.txt") as f_real:
   real_lines = f_real.readlines()

random.shuffle(real_lines)
train_lines = real_lines[0:int(train_split*len(real_lines))]
val_lines = real_lines[len(train_lines):len(train_lines)+int(val_split*len(real_lines))]
test_lines = real_lines[len(train_lines)+int(val_split*len(real_lines)):]


fx = open("label_data/annot_realTrain.txt", "w")
for line in train_lines:
   line = line.rstrip()
   fx.write(line)
   fx.write('\n')
fx.close()
fx = open("label_data/annot_realValidation.txt", "w")
for line in val_lines:
   line = line.rstrip()
   fx.write(line)
   fx.write('\n')
fx.close()
fx = open("label_data/annot_realTest.txt", "w")
for line in test_lines:
   line = line.rstrip()
   fx.write(line)
   fx.write('\n')
fx.close()

with open("label_data/annot_realTrain.txt") as f_train:
   train_lines = f_train.readlines()    
f1 = open('label_data/annot_synthetic.txt')
lines = f1.readlines()
for i in range(len(lines)):
   lines[i] = lines[i].replace(".png", ".jpg", 1)
   #lines[i] = lines[i].replace(" ", " ", 1)
# lines += "\n"
f_synth = open('label_data/annot_synthetic_only.txt','w')
random.shuffle(lines)
f_synth.writelines(lines)
f_synth.close()
lines += train_lines
random.shuffle(lines)
print(lines[0])

f2 = open('label_data/annot_mixed.txt', 'w')
f2.writelines(lines)
f1.close()
f2.close()

# # ct = 0

# # with open(os.getcwd()+"/"+"annotall_shuffc.txt") as f:
# # 	lines = f.readlines()

# # with open(os.getcwd()+"/"+"annot_test.txt") as f:
# # 	lines2 = f.readlines()

# # with open(os.getcwd()+"/"+"annot_val.txt") as f:
# # 	lines3 = f.readlines()


# #with open(os.getcwd()+"/"+'annotall.txt',"a+") as f:
# #    print("Y")
# #    for a, b in zip(train["Location"], train["Annotations"]):
# #        f.write(a+" "+b)
# #        f.write("\n")




