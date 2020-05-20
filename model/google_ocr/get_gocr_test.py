import os
from natsort import ns, natsorted
from os import listdir
from os.path import isfile, join
mypath = os.getcwd()+"/model/google_ocr/gocr_test"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
s = []
ft = open(os.getcwd()+"/model/google_ocr/gocr_out.txt","w")
files = natsorted(onlyfiles, key=lambda y: y.lower())
for fi in files:
    with open(mypath+'/'+fi) as f:
        x = f.readlines()
    if len(x)>2:
        ft.write(x[2].strip())
        ft.write('\n')
    else:
        s.append(ct)
        ft.write("\n")

print("Test lines with no gocr outputs: ", s)


ft.close()
