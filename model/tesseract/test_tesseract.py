import cv2
import sys
import os
import pandas as pd
import numpy as np
import pytesseract

if len(sys.argv)<2:
    sys.exit("format python model/tesseract/test_tesseract.py path_to_test_file")
DIRECTORY_PATH = os.getcwd()
path_to_testfile = sys.argv[1]
config = ('-l Devanagari --psm 7')
# print(DIRECTORY_PATH)
txtfile = os.getcwd()+'/model/tesseract/'+"tesseract_test.txt"
f2 = open(txtfile,"w")
ct = 0
l=0
with open(os.getcwd()+'/'+path_to_testfile) as f:
    lines = f.readlines()
for line in lines:
    loc, annot = line.split("\t",1)[0] , line.split("\t",1)[1]
    imgfile = loc
    # txtfile = os.getcwd()+'/'+tesseract_test.txt
    # if os.path.exists(txtfile)==False:
    
    im = cv2.imread(imgfile)
    text = pytesseract.image_to_string(im, config=config)
    
    f2.write(text.rstrip())
    f2.write('\n')
    # print(text)
    
f2.close()
