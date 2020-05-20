import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

DIRECTORY = os.getcwd()+'/line_images'

for subdir, dirs, files in os.walk(DIRECTORY):
    for f in files:
        if f.endswith(".jpg") or f.endswith(".png"):
            im = Image.open(subdir+'/'+f)
            gray = im.convert("L")
            gray.save(subdir+'/'+f[0:f.find('.')]+".jpg")
            if f.endswith(".png"):
                gray = plt.imread(subdir+'/'+f[0:f.find('.')]+".jpg")
                rows = set([])
                columns = set([])
                for i in range(gray.shape[0]):
                    for j in range(gray.shape[1]):
                        if gray[i][j]<128:
                            rows.add(i)
                            columns.add(j)
                roi = gray[min(rows):max(rows)+1, min(columns):max(columns)+1]
                im_cvt = Image.fromarray(roi)
                im_cvt.save(subdir+'/'+f[0:f.find('.')]+".jpg")




for subdir, dirs, files in os.walk(DIRECTORY):
    for f in files:
        if f.endswith('.png'):
            os.remove(subdir+'/'+f)