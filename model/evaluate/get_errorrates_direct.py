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


import os
import sys

gt_file = sys.argv[1]
pred_file  = sys.argv[2]

with open(gt_file) as f:
    gt_lines = f.readlines()
gt_lines = [' '.join(x.strip().split()) for x in gt_lines]

print(len(gt_lines))
with open(pred_file) as f:
    pred_lines = f.readlines()
pred_lines = [' '.join(x.strip().split()) for x in pred_lines]

print(len(pred_lines))
cerr = total_cer(gt_lines, pred_lines)
werr = total_wer(gt_lines, pred_lines)
serr = total_ser(gt_lines, pred_lines)

print("CER:", cerr)
print("WER:", werr)
print("SER:", serr)
