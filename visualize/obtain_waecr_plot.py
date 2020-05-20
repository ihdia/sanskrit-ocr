from collections import Counter
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import sys
import os

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 30}

mp.rc('font', **font)

if len(sys.argv) < 5:
	sys.exit("Format python obtain_waecr_plot.py <test_file_path> <predicted_test_file_byaocr_path> <predicted_test_file_by_crnn_path> <actual_crnn_test_file_path>")

actual_test_file = sys.argv[1]
predicted_aocr_file = sys.argv[2]
predicted_crnn_file = sys.argv[3]
actual_crnn_test_file = sys.argv[4]

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def movingaverage(interval, window_size):
	window= np.ones(int(window_size))/float(window_size)
	return np.convolve(interval, window, 'same')

def readFileLinesInList(f2):
	with open(f2) as f:
	    content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	content_preds = [x.strip() for x in content]
	return content_preds

def extract_Fractional_plots(f1,f2):
	with open(f1) as f:
	    content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	content_labels = [x.strip().split(".jpg ")[1] for x in content] 
	content_fnames = [x.strip().split(".jpg ")[0] for x in content]
	gts = []
	content_preds = readFileLinesInList(f2)
	errors = dict()
	wordlens = dict()
	for i in range(len(content)):
		if True:
			gts.append(content_labels[i])
			pred_list = content_preds[i].split()
			for word in content_labels[i].split():
				if word not in pred_list:
					#if (len(word) >= 50): print("1 incorrect attnlstm word:", word, ", len:", len(word) ,", preds:", content_preds[i]," gt:", content_labels[i],", fname:", content_fnames[i])
					#if (len(word) == 1): print("1 incorrect attnlstm word:", word, ", preds", content_preds[i]," gt:", content_labels[i],", fname:", content_fnames[i])
					errors[len(word)] = errors.get(len(word), 0) + findmininlist(word,pred_list)
				else:
					#if (len(word) == 1): print("1 correct attnlstm word:", word, ", preds", content_preds[i]," gt:", content_labels[i],", fname:", content_fnames[i])
					pred_list.remove(word)
				wordlens[len(word)] = wordlens.get(len(word), 0) + 1
	#print("total words: ", wordlens.get(1, 0), ", erroneous words: ", errors.get(1, 0))
	x = []
	y = []
	for i in sorted (wordlens) : 
		#print ((i, wordlens[i]), end =" ")
		x.append(i)
		y.append(errors.get(i, 0)/(wordlens.get(i, 0)))
	#print("\n")
	y_av = movingaverage(y, 1)
	return x, y, sum(y), gts
def extract_Fractional_plots3(f1,f2, gts):
	with open(f1) as f:
	    content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	content_labels = [x.strip() for x in content] 
	#content_fnames = [x.strip().split(".jpg ")[0] for x in content]
	content_preds = readFileLinesInList(f2)
	errors = dict()
	wordlens = dict()
	for i in range(len(content)):
		if content_labels[i] in gts:
			pred_list = content_preds[i].split()
			for word in content_labels[i].split():
				if word not in pred_list:
					#if (len(word) == 1): print("2 incorrect crnn word:", word, ", preds", content_preds[i]," gt:", content_labels[i])#,", fname:", content_fnames[i])
					errors[len(word)] = errors.get(len(word), 0) + findmininlist(word,pred_list)
				else:
					#if (len(word) == 1): print("2 correct crnn word:", word, ", preds", content_preds[i]," gt:", content_labels[i])#,", fname:", content_fnames[i])
					pred_list.remove(word)
				wordlens[len(word)] = wordlens.get(len(word), 0) + 1
	#print("total words: ", wordlens.get(1, 0), ", erroneous words: ", errors.get(1, 0))
	x = []
	y = []
	ylens = []
	for i in sorted (wordlens) : 
		#print ((i, wordlens[i]), end =" ")
		x.append(i)
		y.append(errors.get(i, 0)/(wordlens.get(i, 0)))
		ylens.append(wordlens[i])
	#print("\n")
	y_av = movingaverage(y, 1)
	return x, y, sum(y), ylens

def findnearistinlist(source1,list1,mid):
	edit_dists = []
	for line in list1:
		edit_dists.append(levenshteinDistance(source1,line))
	indexes = [i for i, x in enumerate(edit_dists) if x == min(edit_dists)]
	if len(indexes) > 1:
		#mid = ((len(list1) + 1)/2) - 1
		if mid in indexes:
			return list1[mid]
		else:
			return list1[indexes[0]]
	else:
		return list1[indexes[0]]

def findmininlist(word,list1):
	edit_dists = []
	if len(list1) == 0: return len(word)
	for word1 in list1:
		edit_dists.append(levenshteinDistance(word,word1))
	return min(edit_dists)

def findnearestinlist(word,list1):
	edit_dists = []
	editlen = 100000
	nrst = ''
	for word1 in list1:
		editnow = levenshteinDistance(word,word1)
		if  editnow < editlen:
			editlen = editnow
			nrst = word1
	return str(nrst)

def extract_Fractional_plots2(f1,f2):
	with open(f1) as f:
	    content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	content_labels = [x.strip().split(".jpg ")[1] for x in content] 
	content_fnames = [x.strip().split(".jpg ")[0] for x in content]
 
	errors = dict()
	wordlens = dict()
	charerr = 0
	worderr = 0
	senterr = 0
	chartotal = 0
	wordtotal = 0
	senttotal = 0
	for i in range(len(content)):
		if True:
			#print(f2 + content_fnames[i].split("/")[-3] + "/" + content_fnames[i].split("/")[-2] + ".jpg.txt")
			predfilelines1 = readFileLinesInList(f2 + content_fnames[i].split("/")[-3] + "/" + content_fnames[i].split("/")[-2] + ".jpg.txt")
			predfilelines = []
			for ln in predfilelines1: 
				if (ln == "\n") or (len(ln.split()) == 0):
					pass
				else:
					predfilelines.append(ln)
			#print(len(predfilelines))
			lineno = int(content_fnames[i].split("/")[-1]) - 1
			#print(content_fnames[i] + "\t" + content_labels[i])
			#print(len(predfilelines), lineno)
			#print(predfilelines)
			if (lineno < len(predfilelines)) and ((levenshteinDistance(content_labels[i],predfilelines[lineno])/len(content_labels[i])) <= 0.2): #(predfilelines[lineno] == content_labels[i]) or
				pred = predfilelines[lineno]
				#print("here", pred)
			else:
				#pred = findnearestinlist(content_labels[i], predfilelines[max(0,lineno-3):min(len(predfilelines),lineno+4)])
				pred = ' '.join(predfilelines[max(0,lineno-3):min(len(predfilelines),lineno+4)])#findnearistinlist(content_labels[i], predfilelines[lineno-1:lineno+2], lineno - len(predfilelines[:lineno-1]))
			#print (content[i],pred)
			print(pred)
			pred_list = pred.split()
			if (' '.join(content_labels[i].split())) in ' '.join(pred_list):
				senterr += 0
			else:
				senterr += 1
			senttotal += 1
			for word in content_labels[i].split():
				wordtotal += 1
				chartotal += len(word)
				if word not in pred_list:
					#if (len(word) == 29): print(word)
					errors[len(word)] = errors.get(len(word), 0) + findmininlist(word,pred_list)
					worderr += 1
					charerr += findmininlist(word,pred_list)
				else:
					pred_list.remove(word)
				wordlens[len(word)] = wordlens.get(len(word), 0) + 1
	x = []
	y = []
	print("c,w,e", 100*charerr/chartotal, 100*worderr/wordtotal, 100*senterr/senttotal)
	ylens = []
	for i in sorted (wordlens) : 
		#print ((i, wordlens[i]), end =" ")
		x.append(i)
		y.append(errors.get(i, 0)/(wordlens.get(i, 0)))
	#print("\n")
	y_av = movingaverage(y, 1)
	return x, y, sum(y)

bar_width = 0.5
opacity = 0.8
 

x, y_av, auc, content_fnames = extract_Fractional_plots(actual_test_file, predicted_aocr_file)
x2, y_av2, auc2, ylens = extract_Fractional_plots3(actual_crnn_test_file, predicted_crnn_file,content_fnames)
y_av2_convinp = [y_av2[0]]*100 + y_av2 + [y_av2[50]]*100
y_av2_conv = movingaverage(y_av2_convinp, 7)
y_av2_conv = y_av2_conv[101:152]
y_av_convinp = [y_av[0]]*100 + y_av + [y_av[50]]*100
y_av_conv = movingaverage(y_av_convinp, 7)
y_av_conv = y_av_conv[101:152]
# Colorize the graph based on likeability:
likeability_scores2 = np.array(y_av2)
likeability_scores = np.array(y_av)
 
data_normalizer = mp.colors.Normalize()
color_map = mp.colors.LinearSegmentedColormap(
    "my_map",
    {
        "red": [(0, 1.0, 1.0),
                (1.0, .5, .5)],
        "green": [(0, 0.5, 0.5),
                  (1.0, 0, 0)],
        "blue": [(0, 0.50, 0.5),
                 (1.0, 0, 0)]
    }
)

#line_up, = plt.bar(x2, y_av2, label='Ind.senz, Area Under Curve =' + str(auc2))
# x3, y_av3, auc3 = extract_Fractional_plots2('label_data/annot_realTest.txt', 'label_data/txts/')
# print(x3)
# print(y_av3)

fig, ax1 = plt.subplots()

# ax1.plot(np.array(x3),y_av3,'blue', linewidth=3, label='Ind.senz [11]')
# ax1.plot(np.array(x3), y_av3, 'o', color='cyan', markersize=1);
#plt.bar(np.array(x2)+0.4, y_av, bar_width,color='lime',label='Our model\'s erroneous OCR Words (in %)')#, Area Under Curve =' + str(sum(y_av2)))
#y_av2[47] = 1# for log scale
ax1.plot(x2,y_av2,'darkorange', linewidth=3, label='Baseline CNN-RNN [34]')
ax1.plot(np.array(x2), y_av2, 'o', color='gold', markersize=1);
print(x2)
print(y_av2)
#y_av[47] = 1# for log scale
ax1.plot(np.array(x),y_av,'green', linewidth=3, label='Attention LSTM')
ax1.plot(np.array(x), y_av, 'o', color='lime', markersize=1);
ax1.set_xlabel('Word Length')
ax1.set_ylabel('WA-ECR')
ax1.plot([],[],'o', color='red',label='Number of words');
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.19),
          ncol=2, framealpha = 0.005)
ax1.grid(linestyle='-', linewidth='0.5')
print(x)
print(y_av)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#plt.bar(x2, y_av2, bar_width,color='gold',label='Ind.senz erroneous OCR Words (in %)')#, Area Under Curve =' + str(sum(y_av2)))
ax2.plot(np.array(x2), ylens, 'o', color='red')#,label='Number of words');
ax2.set_ylabel('Number of words', color='red')
#ax2.legend(prop={'size': 16})
ax2.tick_params(axis='y', labelcolor='red')
#plt.bar(np.array(x)+0.2, y_av,bar_width,label = 'Our Model, Area Under Curve =' + str(sum(y_av)))
#line_down, = plt.bar(x, y_av, bar_width, alpha=opacity, color='g',label = 'Our Model, Area Under Curve =' + str(auc))
#plt.legend()#handles=[line_up, line_down])
#plt.ylabel('Precentage of Incorrect Words')
#plt.xlabel('Word Length')
ax2.set_yscale('log')
ax2.grid(linestyle=':', linewidth='0.5', color='red')
#plt.ylim((-1000,5000))
#plt.title('Word length based error analysis of ind.senz OCR')# on Nirnaya Sindhu and Kavyaprakasha of Mammata (Test set only)
#plt.grid()
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


