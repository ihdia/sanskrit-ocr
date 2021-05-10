import os
import sys
import json

from jiwer import wer

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

if len(sys.argv)<2:
    sys.exit("Format python model/CRNN/get_errorrates_crnn.py filename")




file_ = sys.argv[1]

with open(file_) as f:
    lines = f.readlines()




get_step = []
get_wer = []
get_cer = []
min_wer = 100
corr_cer = 100
corr_step = ""
gt_val = []
out_val = []
best_gt_val = []
best_out_val = []
er = 0

if lines[0].find('Step')!=-1:
    """ If condition will execute when we are validating on multiple models"""
    lines += ["Step: 0000"]

    for k,line in enumerate(lines):
        if line.find("Step: ")!=-1:
            if k!=len(lines)-1:
                get_step.append(line.split()[1])
            if len(gt_val)!=0:
                wer_ = total_wer(gt_val, out_val)
                cer_ = total_cer(gt_val, out_val)
                if wer_ < min_wer:
                    min_wer = wer_
                    corr_cer = cer_
                    corr_step = get_step[len(get_step)-2]
                get_wer.append(wer_)
                get_cer.append(cer_)

            gt_val = []
            out_val = []
        else:
            try:
                x = line.find(":",1)
                pred = line.rfind("label:",1)
                c = line.find("****",1)
                gt_text = line[x+2:c-1]
                # f2.write(gt_text)
                # f2.write('\n')
                pred_text = line[pred+7:].rstrip()
                # f.write(pred_text)
                # f.write('\n')
                gt_val.append(gt_text)
                out_val.append(pred_text)
            except:
                er = er + 1
                print(line)
                print("Line Error")
    print(len(get_step))
    print(len(get_wer))
    print(get_step)
    print(get_wer)
    print(get_cer)
    plt_var = {"step" : get_step, "wer" : get_wer, "cer" : get_cer}
    json_object = json.dumps(plt_var, indent = 4)
    with open("./visualize/output_crnn.json", "w") as outfile:
        outfile.write(json_object)
    print("Best model: "+corr_step+" with WER: "+str(min_wer)+" and CER: "+str(corr_cer))
    with open("./model/CRNN/logs/error_rates.txt","w") as er_rate:
        er_rate.write("Best model: "+corr_step+" with WER: "+str(min_wer)+" and CER: "+str(corr_cer))
        er_rate.write("\n")

else:
    """Else condition will execute when we are testing on a single model"""
    ft = open("./model/CRNN/logs/crnn_pred.txt","w")
    ft2 = open("./model/CRNN/logs/crnn_gt.txt","w")


    out_val = []
    gt_val = []
    for k,line in enumerate(lines):
        try:
            x = line.find(":",1)
            pred = line.rfind("label:",1)
            c = line.find("****",1)
            gt_text = line[x+2:c-1]
            ft2.write(gt_text)
            ft2.write('\n')
            pred_text = line[pred+7:].rstrip()
            ft.write(pred_text)
            ft.write('\n')
            gt_val.append(gt_text)
            out_val.append(pred_text)
        except:
            er = er + 1
            print(line)
            print("Line Error")
    wer_ = total_wer(gt_val, out_val)
    cer_ = total_cer(gt_val, out_val)
    print("WER on test data: ", wer_)
    print("CER on test data: ", cer_)
    ft.close()
    ft2.close()
