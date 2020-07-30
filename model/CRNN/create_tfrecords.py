#!/usr/bin/python

import re
import os
import sys
import cv2
from tqdm import tqdm
from os import walk
import random
from PIL import Image, ImageFile
import numpy as np
import skimage.io as io
import logging
import tensorflow as tf
ImageFile.LOAD_TRUNCATED_IMAGES = True

inputFile = sys.argv[1]
saving_dir = sys.argv[2]
filename_pref_no_path = os.path.split(inputFile)[1]

def int64_feature(value):
    """
        Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    value_tmp = []
    is_int = True
    for val in value:
        if not isinstance(val, int):
            is_int = False
            value_tmp.append(int(float(val)))
    if is_int is False:
        value = value_tmp
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature(value):
    """
        Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    value_tmp = []
    is_float = True
    for val in value:
        if not isinstance(val, int):
            is_float = False
            value_tmp.append(float(val))
    if is_float is False:
        value = value_tmp
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_feature(value):
    """
        Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, bytes):
        if not isinstance(value, list):
            value = value.encode('utf-8')
        else:
            value = [val.encode('utf-8') for val in value]
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

import sys
sys.path.append(os.getcwd()+'/model/CRNN/local_utils')
import establish_char_dict
char_list = establish_char_dict.CharDictBuilder.read_char_dict(os.getcwd()+'/'+"model/CRNN/data/char_dict/char_dict.json")
ord_map = establish_char_dict.CharDictBuilder.read_ord_map_dict(os.getcwd()+'/'+"model/CRNN/data/char_dict/ord_map.json")
def char_to_int(char):
    """

    :param char:
    :return:
    """
    temp = ord(char)
    # convert upper character into lower character
    if 65 <= temp <= 90:
        temp = temp + 32

    for k, v in ord_map.items():
        if v == str(temp):
            temp = int(k)
            break

    # TODO
    # Here implement a double way dict or two dict to quickly map ord and it's corresponding index

    return temp

def int_to_char(number):
    """

    :param number:
    :return:
    """
    if number == '1':
        return '*'
    if number == 1:
        return '*'
    else:
        return char_list[str(number)]

def read_charset(filename, null_character=u'\u2591'):
	pattern = re.compile(r'(\d+)\t(.+)')
	charset = {}
	with tf.gfile.GFile(filename) as f:
		for i, line in enumerate(f):
			m = pattern.match(line)
			if m is None:
				logging.warning('incorrect charset file. line #%d: %s', i, line)
				continue
			code = int(m.group(1))
			char = m.group(2)#.decode('utf-8')
			if char == '<nul>':
				char = null_character
			charset[char] = code
	return charset
def encode_label(label):
    return [char_to_int(char) for char in label]

def encode_utf8_string(text='abc', charset={'a':0, 'b':1, 'c':2}, length=5, null_char_id=1):
    char_ids_padded = []
    char_ids_unpadded = []
    for i in range(0,len(text)):
        char_ids_unpadded.append(charset[text[i]])
        char_ids_padded.append(charset[text[i]])
    for i in range(len(text),length):
        char_ids_padded.append(null_char_id)
    return char_ids_padded,char_ids_unpadded

def encode_utf8_stringOld(text='abc', charset={'a':0, 'b':1, 'c':2}, length=5, null_char_id=1):
	
	char_ids_padded = null_char_id*np.ones(len(text), dtype=int)
	char_ids_unpadded = null_char_id*np.ones(length, dtype=int)
	for i in range(0,len(text)):
		char_ids_padded[i]= charset[text[i]]
		char_ids_unpadded[i]= charset[text[i]]
	return char_ids_padded.tolist(), char_ids_unpadded.tolist()


#Write code to load image text Path pairs:-
filename_pairs =  []

cnt = 0
Indexlist = []
FileNamelist = []
maxLineLen = 200 
content = ""

with open(inputFile) as f:
	content = f.readlines()

content = [x.strip() for x in content]
for ln in content:
	flname = ln.split(" ",1)[0]#.split("/")[-1]
	if len(ln.split(" ",1)) == 2:
		filename_pairs.append((''+flname ,ln.split(" ",1)[1]))
		maxLineLen = max(maxLineLen, len(ln.split(" ",1)[1]))
		cnt += 1
	elif len(ln.split(" ",1)) == 1:
		cnt += 1
		filename_pairs.append((''+flname ,""))
print(maxLineLen)
charset = read_charset('model/CRNN/charset.txt')

def squarifyRgb(img):
	greyscale = img.convert('L')
	if False: 
		M  = img[:,:,0]
	else:
		M = np.array(greyscale)
	
	# (a,b) = 480, 260

	if False:
		img1 = np.array(Image.fromarray(M).resize((480, 260), Image.ANTIALIAS))
	else:
		img1 = M

	return (img1)

# tfrecords_filename = saving_dir+'/'+'train.tfrecords'
tfrecords_filename =saving_dir
writer = tf.python_io.TFRecordWriter(tfrecords_filename)
cnt1 =  0
cnt2 = 0
until = int(round(1*len(filename_pairs)))
file_cnt = 1
def padimage(prop):
	global file_cnt
	h,w,_ = prop.shape
	flag = 0
	#if (w>400):
		#prop = prop[:,:400,:]
		#h,w,_ = prop.shape
	if (1):#66
		#print(img_path)
		#prop = prop[:,:700,:]
		#h,w,_ = prop.shape
		#hnew = 32#
		hnew  = 32#66
		prop = cv2.resize(prop, (w*hnew//h,hnew))
		h,w,_ = prop.shape
		flag = 1
	max_wd = 1000
	max_ht = 32
	left = (max_wd-w)//2
	# right = res_wd-left

	up = (max_ht-h)//2
	# down = res_ht-up

	# print(left, w)
	# print(up, h)

	max_colour = np.amax(prop)

	xtra = np.ones((max_ht, max_wd, 3))*max_colour
	# print(xtra.shape)

	xtra[0:0+h, 0:0+w] = prop.copy()
	# print(xtra.shape)


	#H,W = 32,600#60, 400#
	#xtra = cv2.resize(xtra, (W,H))
	#print('xtra shape')
	#print(xtra.shape)
	#cv2.imwrite(str("{0:03}".format(file_cnt-1))+'.png', xtra)
	file_cnt += 1
	return np.array(xtra)

print("Writing Until " +str(until))
error_count = 0

for img_path, text in tqdm(filename_pairs[0:until]):
    imagename = os.path.split(img_path)[1]
    print(imagename)
    try:
        cnt1 = cnt1+1
        try:
            prop = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = padimage(prop)
        except Exception as e:
            error_count+=1
            print("Exception for %s, skipping file"%img_path)
            print(type(e).__name__,e)
            continue
        num_of_views = 1
        if len(text)>maxLineLen:
            print(text)
            continue
        print(img.shape)
        # _,img = cv2.imencode('.jpg',img)
        # print(img)
        img = bytes(list(np.reshape(img, [32*1000*3]).astype(np.int64)))
        # print(type(img))
        char_ids_padded, char_ids_unpadded = encode_utf8_string(text=text.replace('\u200c',''), charset = charset, length=maxLineLen, null_char_id = 1)
        # print(char_ids_unpadded)
        example = tf.train.Example(features=tf.train.Features(feature={'labels': int64_feature(encode_label(text.replace('\u200c',''))), 'images': bytes_feature(img), 'imagenames': bytes_feature(imagename)}))
        writer.write(example.SerializeToString())
        cnt2 = cnt2+1
    except Exception as e:
        error_count+=1
        print("Exception for %s, skipping file"%img_path)
        print(type(e).__name__,e)
        continue


print("Tfrecord writing completed with %d errors."%error_count)
writer.close()
