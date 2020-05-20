import json
y = list(',.0123456789-_|# \'~')
CHARMAP = [chr(i) for i in range(2304, 2432)] + y
char_dict = {str(ord(c)): c for c in CHARMAP}

with open('char_dict.json','w',encoding = 'utf-8') as json_f:
    json.dump(char_dict, json_f, sort_keys=True, indent=4)

ord_dict = {str(i):str(ord(c)) for i, c in enumerate(CHARMAP)}
with open('ord_map.json','w',encoding = 'utf-8') as json_f:
    json.dump(ord_dict, json_f, sort_keys=True, indent=4)
