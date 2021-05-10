import os
import sys
import json
from jiwer import wer


def select(arr,i):
    han4 = [2535,1040, 8221, 2985, 3006, 2792, 8377, 8226, 8220, 8216, 2798, 2965, 3585, 8211, 8212, 1041, 3594, 8204, 7386, 8205, 8211, 8217, 8220,8221]
    han2 = [93, 61, 42, 47, 58, 96, 63, 40, 37, 41, 59, 34, 43, 33,99]

    if out[i:i+5]=='43251' and out[i:i+5]=='65279':
        out_word = out_word+chr(int(out[i:i+5]))
    elif arr[i:i+2]=='23' or arr[i:i+2]=='24' or int(arr[i:i+4]) in han4: #10,82,29,30,27,83,35
        return 4
    elif arr[i:i+3] == '124' or arr[i:i+3] == '183':
        return 3

    elif arr[i:i+2]<='96' and arr[i:i+2]>='32':
        return 2
    else:
        return -1
    

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


def calculate_wer(gt,pred):
    return wer(gt, pred)


def total_cer(gt_lines, pred_lines):
    cer_sum = 0
    length_line = 0
    for gt, pred in zip(gt_lines, pred_lines):
        cer_line = levenshteinDistance(gt, pred)
        # print(cer_line)
        cer_sum = cer_sum + cer_line
        length_line = length_line+ len(gt)
    return 100*cer_sum/length_line

def total_wer(gt_lines, pred_lines):
    wer_sum = 0
    total_words = 0
    for gt, pred in zip(gt_lines, pred_lines):
        wer_line = calculate_wer(gt, pred)
        words = len(gt.split())
        total_words = total_words + words
        wer_sum  = wer_sum + (wer_line*words)
    return 100*wer_sum/total_words


def total_ser(gt_lines, pred_lines):
    ser_sum = 0    
    total_lines = 0
    for gt, pred in zip(gt_lines, pred_lines):
        err = 0
        if gt == pred:
            err = 0
        else:
            err = 1
        total_lines += 1
        ser_sum += err
    return 100*ser_sum/total_lines

if len(sys.argv)< 2:
    print('Usage : python model/evaluate/get_errorates.py <path_to_predicted_txt_file>')
fileName = sys.argv[1]
try:
    with open(fileName, "r") as f:
        lines = f.readlines()
except:
    print('Either '+fileName+' does not exist or you have provided the wrong path. Kindly re-check.')
# print(lines)
s=0
get_step = []
get_wer = []
get_cer = []
get_loss = []
sum_loss = 0.0
min_wer = 100
corr_cer = 100
corr_step = ""
gt_val = []
out_val = []
best_gt_val = []
best_out_val = []
ct = 0
x=0
total_acc = loss = '' 
if lines[0].find('Step')!=-1:
    """ If condition will execute when we are validating on multiple models"""
    lines += ["Step: 0000"]
    for k,line in enumerate(lines):
        if line.find('Step: ')!=-1:
            if k!=len(lines)-1:
                get_step.append(line.split()[1])
            if len(gt_val)!=0:
                # get_wer.append(s/ct)
                # get_cer.append(1-float(total_acc))
                wer_ = total_wer(gt_val, out_val)
                cer_ = total_cer(gt_val, out_val)
                if wer_<min_wer:
                    min_wer = wer_
                    corr_cer = cer_
                    corr_step = get_step[len(get_step)-2]
                get_wer.append(wer_)
                get_cer.append(cer_)
                get_loss.append(float(sum_loss/ct))
                

            ct = 0
            gt_val = []
            out_val = []
            sum_loss = 0


        else:
            try:
                out = line.split()[0]
                gt = line.split()[1]

                acc = line.split()[2]
                loss = line.split()[3]    
                total_acc = line.split()[4]
                words = []
                sum_loss = sum_loss + float(loss)
            
                ct = ct+1
                out_word=""
                i = 0
                while i < len(out):
                    shift = select(out,i)
                    if (shift!=-1):
                        out_word = out_word + chr(int(out[i:i+shift]))
                        i = i+shift
                    else:
                        i = i+1
                        out_word = out_word + " "
                       
                
                
                gt_word = ""
                i = 0
                while i < len(gt):
                    gtshift = select(gt,i)
                    if (gtshift!=-1):
                        gt_word = gt_word + chr(int(gt[i:i+gtshift]))
                        i = i + gtshift
                    else:
                        i = i+1
                        out_word = out_word + " "
                        
                
                gt_val.append(gt_word)
                out_val.append(out_word)
        
            except:
                x=  x+1
                print(line)
                print("Line Error")
    print(len(get_step))
    print(len(get_wer))
    print(get_step)
    print("WER - list",get_wer)
    print("CER - list",get_cer)
    print("Loss - list",get_loss)
    plt_var = {"loss" : get_loss,"step" : get_step, "wer" : get_wer, "cer" : get_cer}
    json_object = json.dumps(plt_var, indent = 4)
    with open("./visualize/output.json", "w") as outfile:
        outfile.write(json_object)
    print("Best model: "+corr_step+" with WER: "+str(min_wer)+" and CER: "+str(corr_cer))
    with open("./model/attention-lstm/logs/error_rates.txt","w") as er_rate:
        er_rate.write("Best model: "+corr_step+" with WER: "+str(min_wer)+" and CER: "+str(corr_cer))
        er_rate.write("\n")


else:
    """Else condition will execute when we are testing on a single model"""
    ft = open("./model/attention-lstm/logs/test_gt.txt","w")
    ft2 = open("./model/attention-lstm/logs/test_pred.txt","w")

    out_val = []
    gt_val = []
    for k,line in enumerate(lines):
        try:
            out = line.split()[0]
            gt = line.split()[1]

            acc = line.split()[2]
            loss = line.split()[3]    
            total_acc = line.split()[4]
            words = []
            sum_loss = sum_loss + float(loss)
        
            ct = ct+1
            out_word=""
            i = 0
            while i < len(out):
                shift = select(out,i)
                if (shift!=-1):
                    out_word = out_word + chr(int(out[i:i+shift]))
                    i = i+shift
                else:
                    i = i+1
                    out_word = out_word + " "
                    
            gt_word = ""
            i = 0
            while i < len(gt):
                gtshift = select(gt,i)
                if (gtshift!=-1):
                    gt_word = gt_word + chr(int(gt[i:i+gtshift]))
                    i = i + gtshift
                else:
                    i = i+1
                    out_word = out_word + " "
                    
            out_val.append(out_word)
            gt_val.append(gt_word)
                
            ft.write(gt_word)
            ft2.write(out_word)
            ft.write('\n')
            ft2.write('\n')
            
        
        except:
            x=  x+1
            print("Line Error")
    wer_ = total_wer(gt_val, out_val)
    cer_ = total_cer(gt_val, out_val)
    print("WER on test data: ", wer_)
    print("CER on test data: ", cer_)
    ft.close()
    ft2.close()



