import os
import sys
import pandas as pd
import numpy as np

s="0123456789abcdef"
s_split = list(s)

def next_unicode(s):
    i = len(s)-1
    if s[i]!="f":
        ind = s_split.index(s[i])
        s = s[:i]+s_split[(ind+1)%16]
        return s
    else:
        while s[i]=="f" and i!=0:
            ind = s_split.index(s[i])
            if i==len(s)-1:
                s=s[:i]+s_split[(ind+1)%16]
            else:
                s=s[:i]+s_split[(ind+1)%16]+s[i+1:]
            i=i-1
        if i==0:
            return "$"
        else:
            s = s[:i]+s_split[(s_split.index(s[i])+1)%16]+s[i+1:]
            return s

def if_devanagari(annot):
    ans = False
    for c in annot:
        unicod = str(c.encode("unicode_escape"))
        if unicod[len(unicod)-5:len(unicod)-1]>='0900' and unicod[len(unicod)-5:len(unicod)-1]<='097F':
            return True
    return False



start = "0900"
end = "0980"
res = u''
with open(os.getcwd()+"/DevangariCharList.txt", 'w') as f:
    s = ord(u'\u0900')
    e = ord(u'\u097F') 
    while s<=e:
        f.write(chr(s))
        f.write('\n')
        s=s+1   

    
