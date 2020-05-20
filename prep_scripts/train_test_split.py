#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np 
import os
import random
import time
import sys
import math

import os
import pandas as pd
import numpy as np

DIRECTORY_PATH = os.getcwd()+"/csv_data"

df = pd.DataFrame(columns=['Location','Annotations'])

url_ct = 1
for subdir, dirs, files in os.walk(DIRECTORY_PATH):
   index = subdir.rfind('/')
   book = subdir[index+1:]
   for f in files:
      if f.endswith('_linec.csv'):
         df2 = pd.read_csv(subdir+'/'+f)
         if df2.empty==False:
            for a, b in zip(df2["URL Name"], df2["Annotations"]):
               if os.path.exists(os.getcwd()+'/line_images/'+book+"/"+f[0:f.find('_')]+'/'+str(a)+".jpg"):
                  df.loc[url_ct-1] = [os.getcwd()+'/line_images/'+book+"/"+f[0:f.find('_')]+'/'+str(a)+".jpg", b]
                  url_ct = url_ct + 1
df.to_csv(os.getcwd()+"/label_data/real_annot.csv", index=False)
#Perform this shuffle the dataset
df = pd.read_csv(os.getcwd()+"/label_data/"+'real_annot.csv')
print(df.shape)
df = df.sample(frac=1).reset_index(drop=True)
print(df.shape)
df.to_csv(os.getcwd()+"/label_data/"+"real_annot_sampled.csv", index=False)

df = pd.read_csv(os.getcwd()+"/label_data/"+"real_annot_sampled.csv")

leng = df.shape[0]
print(leng)
train_length =int(0.70*leng)
test_length = int(0.20*leng)
val_length = leng-train_length-test_length

train = df.iloc[0:train_length]
test = df.iloc[train_length:train_length+test_length]
val = df.iloc[train_length+test_length:]



with open("label_data/annot_realTrain.txt","w") as f_train:
	for a, b in zip(train["Location"], train["Annotations"]):
		f_train.write(a+" "+b.strip())
		f_train.write("\n")
	
with open("label_data/annot_realTest.txt","w") as f_test:
   for a, b in zip(test["Location"], test["Annotations"]):
       f_test.write(a+" "+b.strip())
       f_test.write("\n")

with open("label_data/annot_realValidation.txt","w") as f_val:
   for a, b in zip(val["Location"], val["Annotations"]):
       f_val.write(a+" "+b.strip())
       f_val.write("\n")

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




