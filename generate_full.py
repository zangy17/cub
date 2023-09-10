import json
with open('new_labo.json','r') as fp:
    cub=json.load(fp)
full_cub = {}
for c in cub:
    full_cub[c] = []
    for t in cub[c]:
        full_cub[c].append('A photo of a '+c+', which has '+t)
with open('full_labo.json','w') as fw:
    json.dump(full_cub,fw)
