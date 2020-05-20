from PIL import Image
from tesserocr import PyTessBaseAPI, RIL
import cv2
import sys
import os
import numpy as np
import pytesseract
from pytesseract import Output
import logging
import numpy as np
import shutil
import base64
import requests
from natsort import natsorted, ns
import json
import time
from subprocess import Popen

if len(sys.argv)<2:
	sys.exit("Format python samples_pred.py <page or line image>")

typ = sys.argv[1]
time.sleep(5)

DIRECTORY_PATH = os.getcwd()+"/samples/"
URL = "http://localhost:9001/v1/models/sanskrit:predict"
headers = {"cache-control": "no-cache","content-type": "application/json"}
  
# os.system("tensorflow_model_server --port=9000 --rest_api_port=9001 --model_name=sanskrit --model_base_path="+os.getcwd()+'/'+model_dir)
def image_resize(image, width = None, height = 32, inter = cv2.INTER_CUBIC):
	dim = None
	(h, w) = image.shape[:2]
	resized = image
	if width is None and height is None:
		return image
	if width is None:
		if h>32:
			r = height / float(h)
			dim = (int(w * r), height)
			resized = cv2.resize(image, dim, interpolation = inter)
	else:
		r = width / float(w)
		dim = (width, int(h * r))
	
	return resized

def output_value(image_path):
	image_content = base64.b64encode(open(image_path,'rb').read()).decode('utf-8')
	body = {"signature_name": "serving_default","inputs": [{"b64":image_content}]}
	r = requests.post(URL, data=json.dumps(body), headers = headers)
	r_json = json.loads(r.text)
	out = r_json['outputs']['output']
	print(image_path[image_path.rfind('/')+1:]+" "+out)
	out_word=""
	i = 0
	while i < len(out):
		if out[i:i+2]=='23' or out[i:i+2]=='24':
			out_word = out_word+chr(int(out[i:i+4]))
			i = i+4
		elif out[i:i+2]=='32' or out[i:i+2]=='35' or out[i:i+2]=='95' or out[i:i+2]=='46' or out[i:i+2]=='44' or out[i:i+2]=='45' or (out[i:i+2]<='57' and out[i:i+2]>='48'):
			out_word = out_word+chr(int(out[i:i+2]))
			i = i+2
		elif out[i:i+3]=='124':
			out_word = out_word+chr(int(out[i:i+3]))
			i = i+3
		else:
			break
	return out_word

if typ=="page":
	if os.path.exists(os.getcwd()+'/sampless'):
		shutil.rmtree(os.getcwd()+'/sampless')
	for subdir, dirs, files in os.walk(DIRECTORY_PATH):
		for f in files:
			imgfile = subdir+'/'+f
			image = Image.open(imgfile)
			image = image.convert('L')
			img = np.array(image)
	
			if not os.path.exists(os.getcwd()+'/sample_outputs'):
				os.mkdir(os.getcwd()+'/sample_outputs')
			if not os.path.exists(os.getcwd()+"/sampless"):
				os.mkdir(os.getcwd()+"/sampless")
			ft = open('sample_outputs/'+imgfile[imgfile.rfind('/')+1:imgfile.find('.jpg')]+'.txt','w')
			cc = 0
			with PyTessBaseAPI(lang='Devanagari') as api:
				api.SetImage(image)
				boxes = api.GetComponentImages(RIL.TEXTLINE, True)
				print('Found {} textline image components.'.format(len(boxes)))
				for i, (im, box, _, _) in enumerate(boxes):
					api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
					ocrResult = api.GetUTF8Text()
					conf = api.MeanTextConf()
					x,y,w,h = box['x'], box['y'], box['w'], box['h']
					# x = int(round(x-0.03*w))
					# y = int(round(y-0.03*h))
					# w = int(round(w+2*0.03*w))
					# h = int(round(h+2*0.03*h))
					roi = img[max(0,y):max(0,y)+h, max(0,x):max(0,x)+w]
					if np.size(roi)!=0:
						z = image_resize(roi, height=32)
						cv2.imwrite("sampless/"+imgfile[imgfile.rfind('/')+1:imgfile.find('.jpg')]+"_"+str(cc)+".jpg", z)
						cc = cc + 1

			dirpath = os.getcwd()+'/sampless'
			filnames = os.listdir(dirpath)
			filenames = natsorted(filnames, key=lambda y: y.lower())
			print(filenames)
			for filename in filenames:
				out_word = output_value(dirpath+'/'+filename)					
				ft.write(out_word)
				ft.write("\n")
			ft.close()

			if os.path.exists(os.getcwd()+'/sampless'):
				shutil.rmtree(os.getcwd()+'/sampless')
	print("Completed. Press Ctrl+C to exit")


if typ=="line":
	if os.path.exists(os.getcwd()+'/sampless'):
		shutil.rmtree(os.getcwd()+'/sampless')
	os.mkdir(os.getcwd()+"/sampless")
	if not os.path.exists(os.getcwd()+'/sample_outputs'):
		os.mkdir(os.getcwd()+'/sample_outputs')
	for subdir, dirs, files in os.walk(DIRECTORY_PATH):
		for f in files:
			imgfile = subdir+'/'+f
			image = Image.open(imgfile)
			image = image.convert('L')
			img = np.array(image)
			z = image_resize(img, height=32)
			cv2.imwrite(os.getcwd()+"/sampless/"+f, z)
			# shutil.copy(subdir+'/'+f, os.getcwd()+'/'+'sampless')
			ft = open('sample_outputs/'+f[0:f.rfind('.')]+'.txt','w')
			out_word = output_value(os.getcwd()+"/sampless/"+f)
		
			ft.write(out_word)
			ft.write("\n")
			ft.close()
			if os.path.exists(os.getcwd()+'/'+'sampless'+'/'+f):
				os.remove(os.getcwd()+'/'+'sampless'+'/'+f)
	if os.path.exists(os.getcwd()+'/sampless'):
		shutil.rmtree(os.getcwd()+'/sampless')
			

	print("Completed. Press Ctrl+C to exit")
