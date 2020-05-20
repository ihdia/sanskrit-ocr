import os

DIRECTORY = os.getcwd()+'/books/books_Shobhika'

for subdir, dirs, files in os.walk(DIRECTORY):
    for f in files:
        if f.endswith('.jpg'):
            os.remove(subdir+'/'+f)