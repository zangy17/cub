import json
with open('CuPL_image_prompts.json','r') as fp:
    o = json.load(fp)
new = {}
for n in o:
    new[n] = [t.lower().replace('a '+n,'This object') for t in o[n][:10]]
    new[n] = [t.replace('an ' + n, 'This object') for t in new[n]]
    new[n] = [t.replace(n, 'These object') for t in new[n]]
with open('no_name.json','w') as fw:
    json.dump(new,fw)
